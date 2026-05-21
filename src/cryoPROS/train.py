import argparse
import sys
from importlib import resources
from . import __version__, logger, options

def parse_argument():
    parser = argparse.ArgumentParser(description = 'Training a conditional VAE deep neural network model from an input initial volume and raw particles with given imaging parameters.')
    json_path = str(resources.files(options) / 'train.json')

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
    args = parse_argument()
    args.gpu_ids = None

    import json
    from .utils import option
    from . import models

    opt = option.parse(args)
    option.mkdirs(opt)
    if models.is_main_process():
        option.save(opt)
        logger.info('training options:' + json.dumps(opt, indent = 2))


    # Random seed
    import random
    import numpy as np
    import torch

    seed = opt['train'].get('manual_seed', None)
    if seed is not None:
        if models.is_main_process():
            logger.info(f'Setting up random seed: {seed}')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    # Initialize model
    model = models.HVAEModel(opt)
    if models.is_main_process():
        logger.info(f'Network architecture: {model.info_network()}')
    model.init_train()
    if models.is_main_process():
        logger.info(f'Network parameter info: {model.info_params()}')


    # Dataset and DataLoader
    from .dataset import ParticleDataset
    from math import ceil
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    for phase, dataset_opt in opt['datasets'].items():
        if phase != 'train':
            raise NotImplementedError(f'Phase [{phase}] is not recognized')

        train_set = ParticleDataset(dataset_opt)
        train_sampler = DistributedSampler(train_set, shuffle = dataset_opt.get('dataloader_shuffle', False)) if models.is_distributed() else None
        train_loader = DataLoader(
            train_set,
            batch_size = dataset_opt.get('dataloader_batch_size', None),
            shuffle = dataset_opt.get('dataloader_shuffle', False) if train_sampler is None else False,
            sampler = train_sampler,
            num_workers = dataset_opt.get('dataloader_num_workers', None),
            drop_last = True,
            pin_memory = True
        )
        train_size = int(ceil(len(train_loader)))
        if models.is_main_process():
            logger.info(f'Number of train images: {len(train_set)}, iters per epoch per process: {train_size}')


    # Train
    import cv2
    from pathlib import Path
    current_epoch = 0
    current_step = 0
    max_iter = opt['train'].get('max_iter', 30000)

    try:
        while True:
            current_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(current_epoch)

            for train_data in train_loader:
                current_step += 1
                model.feed_data(train_data)
                model.optimize_parameters(current_step)
                model.update_learning_rate()

                if models.is_main_process():
                    if current_step % opt['train'].get('checkpoint_print', 1000) == 0:
                        logs = model.current_log()
                        message = f'<epoch:{current_epoch:3d}, iter:{current_step:8,d}, lr:{model.current_learning_rate():.3e}> '
                        for k, v in logs.items():
                            message += f'{k}: {v:.3e} '
                        logger.info(message)

                    if current_step % opt['train'].get('checkpoint_save', 10000) == 0:
                        logger.info('Saving the model')
                        model.save(current_step)

                    if current_step % opt['train'].get('checkpoint_test', 10000) == 0:
                        for i in range(opt['num_gen']):
                            model.test()
                            visuals = model.current_visuals()
                            img = visuals['img_G']

                            img = (img - img.min()) / (img.max() - img.min())
                            img = img.data.squeeze().clamp_(0, 1).numpy()
                            img = np.around(img * 255).astype(np.uint8)
                            assert img.ndim == 2

                            img_path = Path(opt['path']['images']) / f'{i + 1:04d}_{current_step}_G.png'
                            cv2.imwrite(str(img_path), img)

                if current_step > max_iter:
                    break

            if current_step > max_iter:
                break

        if models.is_main_process():
            logger.info('Saving the final model')
            model.save('latest')
            logger.info('End of training')

    finally:
        model.cleanup()

if __name__ == '__main__':
    main()
