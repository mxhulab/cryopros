import argparse
import sys
from importlib import resources
from . import __version__, options
from .logger import configure_logging, emit_progress, logger

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
        '--apix',
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
        '--data_dir',
        default = '.',
        help = 'directory of input raw particle stacks'
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
        '--num_epoch',
        type = int,
        default = 1,
        help = 'number of epochs'
    )

    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()

def _run():
    # Setup options, directories
    args = parse_argument()

    import json
    from .utils import option

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
    from torch.utils.data import DataLoader

    for phase, dataset_opt in opt['datasets'].items():
        if phase != 'train':
            raise NotImplementedError(f'Phase [{phase}] is not recognized')

        train_set = ParticleDataset(dataset_opt)
        train_loader = DataLoader(
            train_set,
            batch_size = dataset_opt['dataloader_batch_size'],
            shuffle = dataset_opt.get('dataloader_shuffle'),
            num_workers = dataset_opt['dataloader_num_workers'],
            drop_last = True,
            pin_memory = True
        )


    # Initialize model
    if not torch.cuda.is_available():
        raise RuntimeError('Require GPU to perform cryopros-recondismic')

    from .models import ReconModel
    model = ReconModel(opt)
    logger.info('Network architecture:' + model.info_network())
    model.init_train()
    logger.info('Network parameter info:' + model.info_params())


    # Train
    import mrcfile
    from pathlib import Path

    current_step = 0
    num_epoch = opt['train'].get('num_epoch', 1)
    batch_size = dataset_opt['dataloader_batch_size']
    num_batch = len(train_loader)
    total_steps = num_epoch * max(1, num_batch)
    logger.info(f'Number of train images: {len(train_set)}, epoch: {num_epoch}, iterations per epoch: {num_batch}')
    emit_progress(
        'cryopros-prep',
        'setup',
        0,
        total_steps,
        'optimization-iteration',
        metadata={'totalEpochs': num_epoch, 'iterationsPerEpoch': num_batch, 'particleCount': len(train_set)},
    )

    checkpoint_print = max(1, min((num_batch + 4) // 5, (1000 + batch_size - 1) // batch_size))
    if 'checkpoint_print' in opt['train']:
        checkpoint_print = min(checkpoint_print, opt['train']['checkpoint_print'])
    checkpoint_test = opt['train'].get('checkpoint_test', 5000)

    for i_epoch in range(num_epoch):
        for i_batch, train_data in enumerate(train_loader):
            current_step += 1
            model.feed_data(train_data)
            model.optimize_parameters(current_step)
            model.update_learning_rate()

            if current_step % checkpoint_print == 0 or i_batch + 1 == num_batch:
                logs = model.current_log()
                message = f'[Epoch {i_epoch + 1}/{num_epoch}][Iter {i_batch + 1}/{num_batch}] step: {current_step}, lr: {model.current_learning_rate():.3e}, '
                for k, v in logs.items():
                    message += f'{k}: {v:.3e}, '
                logger.info(message)
                emit_progress(
                    'cryopros-prep',
                    'reconstruct-micelle',
                    current_step,
                    total_steps,
                    'optimization-iteration',
                    metadata={
                        'epoch': i_epoch + 1,
                        'totalEpochs': num_epoch,
                        'iteration': i_batch + 1,
                        'iterationsPerEpoch': num_batch,
                        'metrics': logs,
                    },
                )

            if current_step % checkpoint_test == 0:
                model.test()
                visuals = model.current_visuals()

                volume = visuals['volume'].numpy().astype(np.float32)
                save_path = Path(opt['path']['models']) / f'{current_step}.mrc'
                with mrcfile.new(save_path, overwrite = True) as mrc:
                    mrc.set_data(volume)
                    mrc.voxel_size = opt['apix']

    logger.info('Saving the final model')
    model.save('latest')
    emit_progress(
        'cryopros-prep',
        'model-saved',
        min(current_step, total_steps),
        total_steps,
        'optimization-iteration',
        checkpoint={'step': current_step, 'name': 'latest'},
    )
    logger.info('End of training')


def main():
    configure_logging(source='cryopros-prep')
    try:
        return _run()
    except Exception:
        logger.exception('CryoPROS micelle reconstruction failed')
        raise

if __name__ == '__main__':
    main()
