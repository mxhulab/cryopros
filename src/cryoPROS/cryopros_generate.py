import numpy as np
import argparse
import mrcfile
import sys
import torch
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from . import __version__
from .utils import generate_uniform_pose, read_para_from_starfile

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
        '--Apix',
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
        default = [0],
        help = 'GPU IDs used for particle generation'
    )

    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()

def main():
    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    args = parse_argument()
    print('Received arguments:', args)

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents = True, exist_ok = True)
    print('Output directory:', str(output_dir))
    assert output_dir, '`--output_path` should be a directory'

    uniform_pose_path = output_dir / f'{args.gen_name}.star'
    stack_path = f'{args.gen_name}.mrcs'
    print('Generating star file & stack file:', str(uniform_pose_path), str(stack_path))
    generate_uniform_pose(args.param_path, uniform_pose_path, stack_path)

    Apix : float = args.Apix
    box_size : int = args.box_size
    rots, trans, ctfs = read_para_from_starfile(uniform_pose_path, Apix, box_size)

    rots = rots.transpose(0, 2, 1) # for cryoDRGN
    r = R.from_matrix(rots).as_quat()
    metas = np.concatenate([r, trans], axis = 1)
    ctfs = ctfs[:, 2:]

    rots = torch.from_numpy(rots).float()
    trans = torch.from_numpy(trans).float()
    ctfs = torch.from_numpy(ctfs).float()
    metas = torch.from_numpy(metas).float()


    # ----------------------------------------
    # Load model
    # ----------------------------------------
    devices = []
    if torch.cuda.is_available():
        num_cuda_devices = torch.cuda.device_count()
        valid_gpu_ids = [gpu_id for gpu_id in args.gpu_ids if 0 <= gpu_id < num_cuda_devices]
        devices = [torch.device(f'cuda:{gpu_id}') for gpu_id in valid_gpu_ids]
        torch.cuda.empty_cache()
    if len(devices) == 0:
        devices = [torch.device('cpu')]
    print('Generating particles by device(s):', devices)

    from .models import HVAE
    states = torch.load(args.model_path)
    models = []
    for device in devices:
        model = HVAE(nf = 64, nls = args.nls, z_dim = 16, box_size = box_size, Apix = Apix, invert = args.invert)
        model.load_state_dict(states, strict = True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad_(False)
        model.to(device)
        models.append(model)

    # ----------------------------------------
    # Generate
    # ----------------------------------------
    num_gen = min(args.num_max, len(rots))
    batch_size : int = args.batch_size
    num_iter = (num_gen + batch_size - 1) // batch_size
    slices = [slice(k * batch_size, min((k + 1) * batch_size, num_gen)) for k in range(num_iter)]
    data_scale : float = args.data_scale
    print(f'Generating {num_gen} particles')

    with mrcfile.new_mmap(output_dir / stack_path, (num_gen, box_size, box_size), mrc_mode = 2, overwrite = True) as mrc:
        mrc.voxel_size = Apix
        offset = 1024 + mrc.header.nsymbt

    # 使用队列来为每个线程初始化它所持有的model和device
    worker_indices = queue.Queue()
    for worker_idx in range(len(devices)):
        worker_indices.put(worker_idx)
    worker_state = threading.local()

    def init_generate_worker():
        worker_idx = worker_indices.get()
        worker_state.model = models[worker_idx]
        worker_state.device = devices[worker_idx]

    def generate_batch(slc):
        model = worker_state.model
        device = worker_state.device

        ctf = ctfs[slc].to(device)
        rot = rots[slc].to(device)
        tran = trans[slc].to(device)
        meta = metas[slc].to(device)

        with torch.no_grad():
            par_gen = model.generate(rotation = rot, trans = tran, ctf_para = ctf, meta = meta)
        par_gen = par_gen.detach().cpu().squeeze().numpy().astype(np.float32) / data_scale
        return par_gen

    with open(output_dir / stack_path, 'r+b') as fout:
        fout.seek(offset)
        with ThreadPoolExecutor(max_workers = len(devices), initializer = init_generate_worker) as executor:
            for par_gen in tqdm(executor.map(generate_batch, slices), total = num_iter, desc = 'Generating particles ...'):
                par_gen.tofile(fout)


if __name__ == '__main__':
    main()
