import os
import torch
import torch.distributed as dist
from collections import OrderedDict
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel import DistributedDataParallel
from ..logger import logger
from .network_hvae import HVAE
from .ddp import local_rank, is_distributed, is_main_process

class HVAEModel(object):
    '''Training/inference wrapper for the cryoPROS HVAE network.

    This class owns the HVAE network, optimizer, scheduler, checkpoint I/O,
    logging, and visualization helpers used by the training pipeline.
    '''

    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt['path']['models']
        self.device = self.init_device()
        self.is_train = opt['is_train']
        self.schedulers = []
        self.model = self.define_net().to(self.device)
        self.continue_skip = 0

        if is_distributed():
            self.model = DistributedDataParallel(
                self.model,
                device_ids = [local_rank()],
                output_device = local_rank(),
            )

    def init_device(self):
        '''Initialize CUDA device and torch distributed process group.
        '''
        if not torch.cuda.is_available():
            raise RuntimeError('Require GPU to perform cryopros-train')

        if is_distributed():
            torch.cuda.set_device(local_rank())
            if not dist.is_initialized():
                backend = 'nccl' if dist.is_nccl_available() else 'gloo'
                dist.init_process_group(backend = backend, init_method = 'env://')
            return torch.device(f'cuda:{local_rank()}')

        return torch.device('cuda')

    def bare_model(self):
        '''Return the underlying network, unwrapped from DDP when needed.
        '''
        return self.model.module if is_distributed() else self.model

    def init_train(self):
        '''Prepare the model, optimizer, scheduler, and log buffer for training.
        '''
        self.opt_train = self.opt['train']
        self.load()
        self.model.train()
        self.define_optimizer()
        self.define_scheduler()
        self.log_dict = OrderedDict()

    def load(self):
        '''Load pretrained network weights when configured.
        '''
        load_path = self.opt['path']['pretrained_net']
        if load_path is not None:
            if is_main_process():
                logger.info(f'Loading model [{load_path}] ...')
            self.load_network(load_path, self.model)

    def save(self, iter_label):
        '''Save the current network weights.
        '''
        if is_main_process():
            self.save_network(self.save_dir, self.bare_model(), iter_label)

    def define_optimizer(self):
        '''Create the Adam optimizer for all model parameters.
        '''
        self.optimizer = Adam(
            self.model.parameters(),
            lr = self.opt_train['optimizer_lr'],
            weight_decay = 0,
        )

    def define_scheduler(self):
        '''Create and register the multi-step learning-rate scheduler.
        '''
        self.schedulers.append(
            MultiStepLR(
                self.optimizer,
                self.opt_train['scheduler_milestones'],
                self.opt_train['scheduler_gamma'],
            )
        )

    def feed_data(self, data):
        '''Move one mini-batch of data to the configured device.
        '''
        self.img = data['img'].to(self.device)
        self.rotation = data['rotation'].to(self.device)
        self.trans = data['trans'].to(self.device)
        self.ctf = data['ctf'].to(self.device)
        self.meta = data['meta'].to(self.device)

    def optimize_parameters(self, current_step):
        '''Run one optimization step and record scalar losses.

        Args:
            current_step (int): Current training step, used for KL annealing and
                skip diagnostics.
        '''
        self.optimizer.zero_grad()

        rec_loss, kl_loss, model_loss = self.model(
            self.img,
            self.rotation,
            self.trans,
            self.ctf,
            self.meta,
        )

        if self.opt_train['KL_anneal'] == 'linear':
            coeff = min(current_step / self.opt_train['KL_anneal_maxiter'], 1)
            kl_weight = coeff * self.opt_train['KL_weight']
        else:
            kl_weight = self.opt_train['KL_weight']

        model_weight = self.opt_train['model_weight']

        loss1 = rec_loss.mean()
        loss2 = kl_weight * kl_loss.mean()
        loss3 = model_weight * model_loss.mean()

        loss = loss1 + loss2 + loss3
        loss.backward()

        loss_nan = torch.any(torch.isnan(loss))
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 200).item()

        if not loss_nan.item() and grad_norm < 180:
            self.optimizer.step()
            self.continue_skip = 0
        else:
            self.continue_skip += 1
            if is_main_process():
                logger.info(f'Current step: {current_step}, num of skip: {self.continue_skip}')

        self.log_dict['Loss'] = loss.item()
        self.log_dict['Reconstruction loss'] = loss1.item()
        self.log_dict['KL loss'] = loss2.item()
        self.log_dict['Model loss'] = loss3.item()

    def test(self):
        '''Generate visualization tensors without gradients.
        '''
        self.model.eval()
        with torch.no_grad():
            self.img_G = self.bare_model().generate(self.rotation, self.trans, self.ctf, self.meta)
        self.model.train()

    def current_log(self):
        return self.log_dict

    def current_visuals(self):
        '''Return visualization tensors on CPU.
        '''
        return OrderedDict(
            img = self.img.detach()[0].float().cpu(),
            img_G = self.img_G.detach()[0].float().cpu(),
        )

    def current_results(self):
        '''Return generated results on CPU.
        '''
        return OrderedDict(
            img = self.img.detach()[0].float().cpu(),
            img_G = self.img_G.detach()[0].float().cpu(),
        )

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def current_learning_rate(self):
        return self.schedulers[0].get_last_lr()[0]

    def print_network(self):
        logger.info(self.describe_network(self.bare_model()))

    def print_params(self):
        logger.info(self.describe_params(self.bare_model()))

    def info_network(self):
        return self.describe_network(self.bare_model())

    def info_params(self):
        return self.describe_params(self.bare_model())

    @staticmethod
    def describe_network(model):
        '''Build a human-readable network architecture summary.

        Args:
            model (torch.nn.Module): Network to describe.

        Returns:
            str: Model class name, number of parameters, and module structure.
        '''
        num_params = sum(param.numel() for param in model.parameters())
        return (
            f'Networks name: {model.__class__.__name__}\n'
            f'Params number: {num_params}\n'
            f'Net structure:\n{model}\n'
        )

    @staticmethod
    def describe_params(model):
        '''Build summary statistics for every tensor in a model state dict.

        Args:
            model (torch.nn.Module): Network whose state tensors will be
                summarized.

        Returns:
            str: Table containing mean, min, max, std, shape, and parameter name.
        '''
        lines = [
            '',
            ' |  mean  |  min   |  max   |  std   |        shape         |  param_name  |',
        ]

        for name, param in model.state_dict().items():
            if 'num_batches_tracked' in name:
                continue

            value = param.data.clone().float()
            lines.append(f' | {value.mean():>6.3f} | {value.min():>6.3f} | {value.max():>6.3f} | {value.std():>6.3f} | {str(tuple(value.shape)):^20} | {name:^12} |')

        return '\n'.join(lines) + '\n'

    @staticmethod
    def save_network(save_dir, model, iter_label):
        '''Save a model state dict to ``save_dir``.

        Args:
            save_dir (str): Directory where the checkpoint is written.
            model (torch.nn.Module): Model to save.
            iter_label (str | int): Label used as ``<iter_label>.pth``.
        '''
        save_path = os.path.join(save_dir, f'{iter_label}.pth')
        states = {key: param.cpu() for key, param in model.state_dict().items()}
        torch.save(states, save_path)

    def load_network(self, load_path, model, strict = True):
        '''Load a model state dict from disk.

        Args:
            load_path (str): Path to a ``.pth`` checkpoint.
            model (torch.nn.Module): Model receiving the state dict.
            strict (bool): Whether checkpoint keys must exactly match the model.
        '''
        states = torch.load(load_path, map_location = self.device)
        model = self.bare_model()
        model.load_state_dict(states, strict = strict)

    def cleanup(self):
        '''Release the distributed process group when this model owns one.
        '''
        if is_distributed() and dist.is_initialized():
            dist.destroy_process_group()

    def define_net(self):
        '''Construct the HVAE network from runtime options.
        '''
        return HVAE(
            nf = self.opt['model']['nf'],
            nls = self.opt['model']['nls'],
            z_dim = self.opt['model']['z_dim'],
            box_size = self.opt['box_size'],
            apix = self.opt['apix'],
            invert = self.opt['invert'],
            init_volume_path = self.opt['init_volume_path'],
            update_volume = self.opt['update_volume'],
            volume_scale = self.opt['volume_scale'],
            noise_level = self.opt['noise_level'],
        )
