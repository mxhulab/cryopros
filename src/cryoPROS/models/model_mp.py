import os
import site
import sys
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel  # , DistributedDataParallel
from collections import OrderedDict
from torch.optim import Adam
from torch.optim import lr_scheduler
site_packages_dir = site.getsitepackages()[0]
package_path = os.path.join(site_packages_dir, "cryoPROS")
sys.path.append(package_path)

class ReconModel():
    def __init__(self, opt):
        self.opt = opt                         # opt
        self.save_dir = opt['path']['models']  # save models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']        # training or not
        self.schedulers = []                   # schedulers
        
        self.model = self.define_net().to(self.device)
        self.model = DataParallel(self.model)
        
    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    def init_train(self):
        self.opt_train = self.opt['train']    # training option
        self.load()                           # load model
        self.model.train()
        self.define_optimizer()               # define optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log 


    def load(self):
        load_path = self.opt['path']['pretrained_net']
        if load_path is not None:
            print('Loading model [{:s}] ...'.format(load_path))
            self.load_network(load_path, self.model)


    def save(self, iter_label):
        self.save_network(self.save_dir, self.model, iter_label)


    def define_optimizer(self):
        optim_params = [p for p in self.model.parameters()]
        self.optimizer = Adam(optim_params, lr=self.opt_train['optimizer_lr'], weight_decay=0)


    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.optimizer,
                                                        self.opt_train['scheduler_milestones'],
                                                        self.opt_train['scheduler_gamma']
                                                        ))

    def feed_data(self, data):
        self.img = data['img'].to(self.device)
        self.rotation = data['rotation'].to(self.device)
        self.trans = data['trans'].to(self.device)
        self.ctf = data['ctf'].to(self.device)
        
        
    def optimize_parameters(self, current_step):
        self.optimizer.zero_grad()
        
        rec_loss = self.model(self.img, self.rotation, self.trans, self.ctf)
        
        loss = rec_loss.mean()
        loss.backward()
                
        self.optimizer.step()
        
        self.log_dict['loss'] = loss.item()
        

    def test(self):
        self.model.eval()
        
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        
        with torch.no_grad():
            self.volume = model.get_volume().clone()
            self.volume_fix = model.volume.clone()

        self.model.train()


    def current_log(self):
        return self.log_dict


    def current_visuals(self):
        out_dict = OrderedDict()
        out_dict['volume'] = self.volume.detach().float().cpu()
        out_dict['volume_fix'] = self.volume_fix.detach().float().cpu()
        return out_dict


    def current_visuals(self):
        out_dict = OrderedDict()
        out_dict['volume'] = self.volume.detach().float().cpu()
        out_dict['volume_fix'] = self.volume_fix.detach().float().cpu()
        return out_dict


    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)


    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]


    """
    # ----------------------------------------
    # Information of net
    # ----------------------------------------
    """

    def print_network(self):
        msg = self.describe_network(self.model)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.model)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.model)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.model)
        return msg

    # ----------------------------------------
    # network name and number of parameters
    # ----------------------------------------
    
    def describe_network(self, model):
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        msg = '\n'
        msg += 'Networks name: {}'.format(model.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(model)) + '\n'
        
        return msg

    # ----------------------------------------
    # parameters description
    # ----------------------------------------
    def describe_params(self, model):
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
        for name, param in model.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.data.clone().float()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
        
        return msg

    """
    # ----------------------------------------
    # Save prameters
    # Load prameters
    # ----------------------------------------
    """
    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, model, iter_label):
        save_filename = '{}.pth'.format(iter_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        model_state_dict = model.state_dict()
        
        for key, param in model_state_dict.items():
            model_state_dict[key] = param.cpu()
        
        states = model_state_dict
        torch.save(states, save_path)


    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, model, strict=True):
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        states = torch.load(load_path)
        model.load_state_dict(states, strict=strict)

    def define_net(self):
        from models.network_mp import Reconstructor
        net = Reconstructor(box_size=self.opt['box_size'], 
                            Apix=self.opt['Apix'], 
                            invert=self.opt['invert'],
                            init_volume_path=self.opt['init_volume_path'], 
                            volume_scale=self.opt['volume_scale'], 
                            update_volume_scale=self.opt['update_volume_scale'], 
                            update_volume=self.opt['update_volume'],
                            mask_path=self.opt['mask_path']
                            )
        return net


