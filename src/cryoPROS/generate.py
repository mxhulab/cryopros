import argparse
import os
import sys
from dataclasses import dataclass
from . import __version__
from .logger import configure_logging, emit_progress, logger


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool

    @property
    def is_main(self):
        return self.rank == 0


def init_distributed(torch):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError('Distributed CryoPROS Generate requires CUDA')
        torch.cuda.set_device(local_rank)
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend = 'nccl')
    return DistributedContext(rank = rank, local_rank = local_rank, world_size = world_size, distributed = distributed)


def barrier(ctx):
    if ctx.distributed:
        import torch.distributed as dist
        dist.barrier()


def destroy_distributed():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def parse_argument():
    parser = argparse.ArgumentParser(description = 'Generating an auxiliary particle stack from a pre-trained conditional VAE deep neural network model.')

    parser.add_argument(
        '-v', '--version',
        action = 'version',
        version = f'%(prog)s {__version__}'
    )
    parser.add_argument(
        '--model_path',
        required = True,
        help = 'input pretrained model path'
    )
    parser.add_argument(
        '--param_path',
        required = True,
        help = 'path of star file which contains the imaging parameters')
    parser.add_argument(
        '--output_path',
        required = True,
        help = 'output output synthesized auxiliary particle stack'
    )
    parser.add_argument(
        '--gen_name',
        required = True,
        help = 'filename of the generated auxiliary particle stack'
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
        '--invert',
        action = 'store_true',
        help = 'invert the image sign'
    )
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 10,
        help = 'batch size'
    )
    parser.add_argument(
        '--num_max',
        type = int,
        default = 100000000,
        help = 'maximum number particles to generate'
    )
    parser.add_argument(
        '--data_scale',
        type = float,
        default = 0.1,
        help = 'scale factor'
    )
    parser.add_argument(
        '--gen_mode',
        type = int,
        default = 2,
        help = '(deprecated) storage model of the synthesized particles; mode 0 is int; mode 2 is float'
    )
    parser.add_argument(
        '--nls',
        type = int,
        nargs = '+',
        default = [2, 2, 4, 4],
        help = 'number of layers of the neural network'
    )
    parser.add_argument(
        '--gpu_ids',
        type = int,
        nargs = '+',
        required = False,
        help = 'GPU IDs used for particle generation; use CPU if not provided'
    )

    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()


def _run():
    # Preparation
    args = parse_argument()
    logger.info(f'Received arguments: {str(args)}')

    import numpy as np
    import torch
    from pathlib import Path
    from .dataset import ParticleDataset
    from .utils import generate_uniform_pose

    ctx = None
    try:
        ctx = init_distributed(torch)
        output_dir = Path(args.output_path)
        if ctx.is_main:
            output_dir.mkdir(parents = True, exist_ok = True)
            logger.info(f'Output directory: {str(output_dir)}')
        barrier(ctx)
        assert output_dir, '`--output_path` should be a directory'

        uniform_pose_path = output_dir / f'{args.gen_name}.star'
        stack_path = f'{args.gen_name}.mrcs'
        if ctx.is_main:
            logger.info(f'Generating star file & stack file: {str(uniform_pose_path)} {str(stack_path)}')
            generate_uniform_pose(args.param_path, uniform_pose_path, stack_path)
        barrier(ctx)

        apix : float = args.apix
        box_size : int = args.box_size
        particle_dataset = ParticleDataset({
            'data_dir': output_dir,
            'param_path': uniform_pose_path,
            'apix': apix,
            'box_size': box_size,
            'n_max': args.num_max,
        })
        rots = particle_dataset.rotations
        trans = particle_dataset.trans
        ctfs = particle_dataset.ctfs
        metas = particle_dataset.metas

        # Load model
        if ctx.distributed:
            gpu_ids = args.gpu_ids or list(range(torch.cuda.device_count()))
            if ctx.local_rank >= len(gpu_ids):
                raise ValueError(f'LOCAL_RANK {ctx.local_rank} has no corresponding GPU ID in {gpu_ids}')
            device = torch.device(f'cuda:{gpu_ids[ctx.local_rank]}')
            torch.cuda.set_device(device)
            devices = [device]
            logger.info(f'[rank {ctx.rank}/{ctx.world_size}] Generating particles on GPU {gpu_ids[ctx.local_rank]}')
        elif args.gpu_ids is None:
            devices = [torch.device('cpu')]
            logger.info('No GPU IDs specified, generating particles by CPU')
        else:
            if not torch.cuda.is_available():
                raise RuntimeError('GPU IDs were specified, but CUDA is not available')
            num_cuda_devices = torch.cuda.device_count()
            invalid_gpu_ids = [gpu_id for gpu_id in args.gpu_ids if not 0 <= gpu_id < num_cuda_devices]
            if invalid_gpu_ids:
                raise ValueError(f'Invalid GPU ID(s): {invalid_gpu_ids}; available GPU ID range is [0, {num_cuda_devices - 1}]')
            devices = [torch.device(f'cuda:{gpu_id}') for gpu_id in args.gpu_ids]
            torch.cuda.empty_cache()
            logger.info(f'Generating particles by GPU(s): {args.gpu_ids}')

        from .models import HVAE
        states = torch.load(args.model_path, map_location = 'cpu')
        models = []
        for device in devices:
            model = HVAE(nf = 64, nls = args.nls, z_dim = 16, box_size = box_size, apix = apix, invert = args.invert)
            model.load_state_dict(states, strict = True)
            model.eval()
            for k, v in model.named_parameters():
                v.requires_grad_(False)
            model.to(device)
            models.append(model)

        # Generate
        num_gen = len(particle_dataset)
        batch_size : int = args.batch_size
        num_iter = (num_gen + batch_size - 1) // batch_size
        slices = [slice(k * batch_size, min((k + 1) * batch_size, num_gen)) for k in range(num_iter)]
        data_scale : float = args.data_scale
        logger.info(f'Generating {num_gen} particles')
        log_interval = max(1, min((num_iter + 4) // 5, (10000 + batch_size - 1) // batch_size))

        def generate_batch(model, device, slc):
            ctf = ctfs[slc].to(device)
            rot = rots[slc].to(device)
            tran = trans[slc].to(device)
            meta = metas[slc].to(device)

            with torch.no_grad():
                par_gen = model.generate(rotation = rot, trans = tran, ctf_para = ctf, meta = meta)
            par_gen = par_gen.detach().cpu().squeeze().numpy().astype(np.float32) / data_scale
            return par_gen.reshape((-1, box_size, box_size))

        import mrcfile
        if ctx.distributed:
            local_start = round(ctx.rank / ctx.world_size * num_gen)
            local_end = round((ctx.rank + 1) / ctx.world_size * num_gen)
            local_slices = [slice(k, min(k + batch_size, local_end)) for k in range(local_start, local_end, batch_size)]
            shard_path = output_dir / f'{args.gen_name}.rank{ctx.rank:04d}.mrcs'
            if local_end > local_start:
                with mrcfile.new_mmap(shard_path, (local_end - local_start, box_size, box_size), mrc_mode = 2, overwrite = True) as mrc:
                    mrc.voxel_size = apix
                    write_offset = 0
                    for k, slc in enumerate(local_slices, start = 1):
                        par_gen = generate_batch(models[0], devices[0], slc)
                        mrc.data[write_offset : write_offset + len(par_gen)] = par_gen
                        write_offset += len(par_gen)
                        if k % log_interval == 0 or k == len(local_slices):
                            logger.info(f'[rank {ctx.rank}] generated particles {local_start + write_offset}/{num_gen}')
                            if ctx.is_main:
                                emit_progress(
                                    'cryopros-generate',
                                    'generate',
                                    local_start + write_offset,
                                    num_gen,
                                    'particle',
                                    metadata={'batch': k, 'rank': ctx.rank, 'worldSize': ctx.world_size},
                                )
            else:
                logger.info(f'[rank {ctx.rank}] no particles assigned')
            barrier(ctx)
            if ctx.is_main:
                with mrcfile.new_mmap(output_dir / stack_path, (num_gen, box_size, box_size), mrc_mode = 2, overwrite = True) as out_mrc:
                    out_mrc.voxel_size = apix
                    for rank in range(ctx.world_size):
                        shard_start = round(rank / ctx.world_size * num_gen)
                        shard_end = round((rank + 1) / ctx.world_size * num_gen)
                        if shard_end <= shard_start:
                            continue
                        shard = output_dir / f'{args.gen_name}.rank{rank:04d}.mrcs'
                        with mrcfile.mmap(shard, permissive = True, mode = 'r') as in_mrc:
                            out_mrc.data[shard_start : shard_end] = in_mrc.data
                        shard.unlink(missing_ok = True)
            barrier(ctx)
        else:
            with mrcfile.new_mmap(output_dir / stack_path, (num_gen, box_size, box_size), mrc_mode = 2, overwrite = True) as mrc:
                mrc.voxel_size = apix
                offset = 1024 + mrc.header.nsymbt

            # 使用队列来为每个线程初始化它所持有的model和device
            import threading
            from queue import Queue
            worker_indices = Queue()
            for worker_idx in range(len(devices)):
                worker_indices.put(worker_idx)
            worker_state = threading.local()

            def init_generate_worker():
                worker_idx = worker_indices.get()
                worker_state.model = models[worker_idx]
                worker_state.device = devices[worker_idx]

            def generate_thread_batch(slc):
                return generate_batch(worker_state.model, worker_state.device, slc)

            from concurrent.futures import ThreadPoolExecutor
            with open(output_dir / stack_path, 'r+b') as fout:
                fout.seek(offset)
                with ThreadPoolExecutor(max_workers = len(devices), initializer = init_generate_worker) as executor:
                    for k, par_gen in enumerate(executor.map(generate_thread_batch, slices), start = 1):
                        par_gen.tofile(fout)
                        if k % log_interval == 0 or k == num_iter:
                            num_done = min(k * batch_size, num_gen)
                            logger.info(f'[{num_done}/{num_gen}] particles generated ({k}/{num_iter} batches)')
                            emit_progress(
                                'cryopros-generate',
                                'generate',
                                num_done,
                                num_gen,
                                'particle',
                                metadata={'batch': k, 'totalBatches': num_iter},
                            )

        if ctx.is_main:
            emit_progress(
                'cryopros-generate',
                'output-written',
                num_gen,
                num_gen,
                'particle',
                metadata={'outputPath': str(output_dir / stack_path)},
            )
            logger.info('Done')
    except Exception:
        logger.exception('CryoPROS generation failed during distributed startup or execution')
        raise
    finally:
        if ctx is not None:
            destroy_distributed()


def main():
    configure_logging(source='cryopros-generate')
    return _run()


if __name__ == '__main__':
    main()
