import json
import os
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

def parse(args):
    # Read a json file with '//' line comment
    opt_path = args.opt
    json_str = ''
    with open(opt_path) as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook = OrderedDict)

    # Write args to opt
    opt['box_size'] = args.box_size
    opt['apix'] = args.apix
    opt['init_volume_path'] = args.init_volume_path
    if hasattr(args, 'mask_path'): opt['mask_path'] = args.mask_path
    opt['invert'] = args.invert
    opt['opt_path'] = args.opt
    opt['task'] = args.task_name
    opt['volume_scale'] = args.volume_scale
    opt['datasets']['train']['data_dir'] = args.data_dir
    opt['datasets']['train']['param_path'] = args.param_path
    opt['datasets']['train']['dataloader_batch_size'] = args.dataloader_batch_size
    opt['datasets']['train']['dataloader_num_workers'] = args.dataloader_num_workers
    opt['train']['optimizer_lr'] = args.lr
    if hasattr(args, 'KL_weight'): opt['train']['KL_weight'] = args.KL_weight
    if hasattr(args, 'num_epoch'):
        opt['train']['num_epoch'] = args.num_epoch
    if hasattr(args, 'max_iter') and args.max_iter is not None:
        opt['train']['max_iter'] = args.max_iter
    opt['is_train'] = True

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['apix'] = opt['apix']
        dataset['box_size'] = opt['box_size']
        if 'data_scale' in opt:
            dataset['data_scale'] = opt['data_scale']

    # path
    path_task = os.path.join(opt['path']['root'], opt['task'])
    opt['path']['task'] = path_task
    opt['path']['log'] = path_task
    opt['path']['options'] = os.path.join(path_task, 'options')
    opt['path']['models'] = os.path.join(path_task, 'models')
    opt['path']['images'] = os.path.join(path_task, 'images')

    return opt

def mkdirs(opt):
    for key, path in opt['path'].items():
        if 'pretrained' not in key:
            Path(path).mkdir(parents = True, exist_ok = True)

def save(opt):
    opt_path = Path(opt['opt_path'])
    opt_name = opt_path.stem + datetime.now().strftime('_%y%m%d_%H%M%S') + opt_path.suffix
    opt_dir = Path(opt['path']['options'])
    with open(opt_dir / opt_name, 'w') as fout:
        json.dump(opt, fout, indent = 2)
