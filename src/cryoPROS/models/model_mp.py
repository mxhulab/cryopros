import os
import torch
from collections import OrderedDict
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from .. import logger
from .network_mp import Reconstructor

class ReconModel(object):
    '''Training/inference wrapper for the cryoPROS reconstruction network.

    This class owns the reconstruction network, optimizer, scheduler, checkpoint
    I/O, logging, and visualization helpers used by the training pipeline.
    '''

    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt['path']['models']
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.model = self.define_net().to(self.device)

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
            logger.info(f'Loading model [{load_path}] ...')
            self.load_network(load_path, self.model)

    def save(self, iter_label):
        '''Save the current network weights.
        '''
        self.save_network(self.save_dir, self.model, iter_label)

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

    def optimize_parameters(self, current_step):
        '''Run one optimization step and record the scalar reconstruction loss.

        Args:
            current_step (int): Current training step. Kept for API compatibility;
                this method does not use it directly.
        '''
        self.optimizer.zero_grad()
        rec_loss = self.model(self.img, self.rotation, self.trans, self.ctf)
        loss = rec_loss.mean()
        loss.backward()
        self.optimizer.step()
        self.log_dict['loss'] = loss.item()

    def test(self):
        '''Evaluate model tensors needed for visualization without gradients.
        '''
        self.model.eval()
        with torch.no_grad():
            self.volume = self.model.get_volume().clone()
            self.volume_fix = self.model.volume.clone()
        self.model.train()

    def current_log(self):
        return self.log_dict

    def current_visuals(self):
        '''Return visualization tensors on CPU.

        Returns:
            OrderedDict: Contains the reconstructed volume and fixed volume
                snapshot, both detached from autograd and converted to float CPU
                tensors.
        '''
        return OrderedDict(
            volume = self.volume.detach().float().cpu(),
            volume_fix = self.volume_fix.detach().float().cpu(),
        )

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def print_network(self):
        logger.info(self.describe_network(self.model))

    def print_params(self):
        logger.info(self.describe_params(self.model))

    def info_network(self):
        return self.describe_network(self.model)

    def info_params(self):
        return self.describe_params(self.model)

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
            ' |  mean  |  min   |  max   |  std   |        shape         |  param_name  |'
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
        model.load_state_dict(states, strict = strict)

    def define_net(self):
        '''Construct the reconstruction network from runtime options.
        '''
        return Reconstructor(
            box_size = self.opt['box_size'],
            Apix = self.opt['Apix'],
            invert = self.opt['invert'],
            init_volume_path = self.opt['init_volume_path'],
            volume_scale = self.opt['volume_scale'],
            update_volume_scale = self.opt['update_volume_scale'],
            update_volume = self.opt['update_volume'],
            mask_path = self.opt['mask_path'],
        )
