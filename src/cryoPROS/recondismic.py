import argparse
import sys
from importlib import resources
from . import __version__, logger, options

def parse_argument():
    parser = argparse.ArgumentParser(description = 'Reconstructing the micelle/nanodisc density map from an input initial volume, a mask volume and raw particles with given imaging parameters.')
    json_path = str(resources.files(options) / 'train_mp.json')

    parser.add_argument(
        '-v', '--version',
        action = 'version',
        version = f'%(prog)s {__version__}'
    )
    parser.add_argument(
        '--box_size',
        type = int,
        required = True,
        help = 'box size'
    )
    parser.add_argument(
        '--Apix',
        type = float,
        required = True,
        help = 'pixel size in Angstrom'
    )
    parser.add_argument(
        '--init_volume_path',
        required = True,
        help = 'input inital volume path'
    )
    parser.add_argument(
        '--mask_path',
        required = True,
        help = 'mask volume path'
    )
    parser.add_argument(
        '--data_path',
        required = True,
        help = 'input raw particles path'
    )
    parser.add_argument(
        '--param_path',
        required = True,
        help = 'path of star file which contains the imaging parameters'
    )
    parser.add_argument(
        '--gpu_ids',
        type = int,
        nargs = '+',
        default = [0],
        help = 'GPU IDs to utilize'
    )
    parser.add_argument(
        '--invert',
        action = 'store_true',
        help = 'invert the image sign'
    )
    parser.add_argument(
        '--opt',
        default = json_path,
        help = 'path to option JSON file'
    )
    parser.add_argument(
        '--task_name',
        default = 'cryoPROS',
        help = 'task name'
    )
    parser.add_argument(
        '--volume_scale',
        type = float,
        default = 50.0,
        help = 'scale factor'
    )
    parser.add_argument(
        '--dataloader_batch_size',
        type = int,
        default = 16,
        help = 'batch size to load data'
    )
    parser.add_argument(
        '--dataloader_num_workers',
        type = int,
        default = 0,
        help = 'number of workers to load data'
    )
    parser.add_argument(
        '--lr',
        type = float,
        default = 1e-4,
        help = 'learning rate'
    )
    parser.add_argument(
        '--KL_weight',
        type = float,
        default = 1e-4,
        help = 'KL weight'
    )
    parser.add_argument(
        '--max_iter',
        type = int,
        default = 30000,
        help = 'max number of iterations'
    )

    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()

def main():
    # Setup options, directories, logger
    import json
    from .utils import option

    args = parse_argument()
    opt = option.parse(args)
    option.mkdirs(opt)
    option.save(opt)
    logger.info('training options:' + json.dumps(opt, indent = 2))


    # Random seed
    import random
    import numpy as np
    import torch

    seed = opt['train'].get('manual_seed', None)
    if seed is not None:
        logger.info(f'Setting up random seed: {seed}')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    # Dataset and DataLoader
    from .dataset import ParticleDataset
    from math import ceil
    from torch.utils.data import DataLoader

    for phase, dataset_opt in opt['datasets'].items():
        if phase != 'train':
            raise NotImplementedError(f'Phase [{phase}] is not recognized')

        train_set = ParticleDataset(dataset_opt)
        train_size = int(ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
        logger.info(f'Number of train images: {len(train_set)}, iters: {train_size}')
        train_loader = DataLoader(
            train_set,
            batch_size = dataset_opt.get('dataloader_batch_size', None),
            shuffle = dataset_opt.get('dataloader_shuffle', None),
            num_workers = dataset_opt.get('dataloader_num_workers', None),
            drop_last = True,
            pin_memory = True
        )


    # Initialize model
    if torch.cuda.is_available() and opt['gpu_ids'] is not None:
        logger.info(f'Specifies GPU ids: {opt["gpu_ids"]}')
        num_cuda_devices = torch.cuda.device_count()
        opt['gpu_ids'] = [gpu_id for gpu_id in opt['gpu_ids'] if 0 <= gpu_id < num_cuda_devices]
        logger.info(f'Valid GPU ids: {opt["gpu_ids"]}')
    if not opt['gpu_ids']:
        raise RuntimeError('Require GPU to perform cryopros-recondismic')

    from .models import ReconModel
    model = ReconModel(opt)
    logger.info('Network architecture:' + model.info_network())
    model.init_train()
    logger.info('Network parameter info:' + model.info_params())


    # Train
    import mrcfile
    from pathlib import Path
    current_epoch = 0
    current_step = 0
    max_iter = opt['train'].get('max_iter', 30000)

    while True:
        current_epoch += 1

        for train_data in train_loader:
            current_step += 1
            model.feed_data(train_data)
            model.optimize_parameters(current_step)
            model.update_learning_rate()

            if current_step % opt['train'].get('checkpoint_print', 500) == 0:
                logs = model.current_log()
                message = f'<epoch:{current_epoch:3d}, iter:{current_step:8,d}, lr:{model.current_learning_rate():.3e}> '
                for k, v in logs.items():
                    message += f'{k}: {v:.3e} '
                logger.info(message)

            if current_step % opt['train'].get('checkpoint_test', 5000) == 0:
                model.test()
                visuals = model.current_visuals()

                volume = visuals['volume'].numpy().astype(np.float32)
                save_path = Path(opt['path']['models']) / f'{current_step}.mrc'
                with mrcfile.new(save_path, overwrite = True) as mrc:
                    mrc.set_data(volume)
                    mrc.voxel_size = opt['Apix']

            if current_step > max_iter:
                break

        if current_step > max_iter:
            break

    logger.info('Saving the final model')
    model.save('latest')
    logger.info('End of training')

if __name__ == '__main__':
    main()
