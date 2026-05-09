import numpy as np
import torch
import argparse
import os
import sys
import mrcfile
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from . import __version__
from .utils import read_ctf_from_starfile, read_pose_from_starfile, generate_uniform_pose

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
        help = 'storage model of the synthesized particles; mode 0 is int; mode 2 is float'
    )
    parser.add_argument(
        '--nls',
        type = int,
        nargs = '+',
        default = [2, 2, 4, 4],
        help = 'number of layers of the neural network'
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
    print(args)

    model_path = args.model_path
    output_path = args.output_path
    batch_size = args.batch_size
    box_size = args.box_size
    Apix = args.Apix
    invert = args.invert
    num_max = args.num_max
    gen_name = args.gen_name
    param_path = args.param_path
    data_scale = args.data_scale
    nls = args.nls
    gen_mode = args.gen_mode

    os.system('mkdir -p ' + output_path)

    uniform_pose_path = os.path.join(output_path, f'{gen_name}.star')
    generate_uniform_pose(param_path, uniform_pose_path, gen_name)

    pose = read_pose_from_starfile(uniform_pose_path, Apix, box_size)
    ctfs = read_ctf_from_starfile(uniform_pose_path, Apix, box_size)

    rotations = pose[0]
    trans = pose[1]

    ctfs = ctfs[:, 2:]

    rotations = rotations.transpose(0, 2, 1) # for cryoDRGN

    r = R.from_matrix(rotations)
    r = r.as_quat()
    metas = np.concatenate([r, trans], axis=1)

    rotations = torch.from_numpy(rotations).float()
    trans = torch.from_numpy(trans).float()
    ctfs = torch.from_numpy(ctfs).float()
    metas = torch.from_numpy(metas).float()

    rotations = rotations[:num_max]
    trans = trans[:num_max]
    ctfs = ctfs[:num_max]
    metas = metas[:num_max]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from .models import HVAE
    model = HVAE(nf = 64, nls = nls, z_dim = 16, box_size = box_size, Apix = Apix, invert = invert)
    states = torch.load(model_path)
    model.load_state_dict(states, strict = True)
    model.eval()

    for k, v in model.named_parameters():
        v.requires_grad = False

    model = model.to(device)

    num_gen = rotations.shape[0]

    print(f"Generating {num_gen} particles.")

    num_iter = num_gen // batch_size if num_gen % batch_size == 0 else num_gen // batch_size + 1

    with mrcfile.new_mmap(os.path.join(output_path, f'{gen_name}.mrcs'), shape=(num_gen, box_size, box_size), mrc_mode=gen_mode, overwrite=True) as mrc:
        for k in tqdm(range(num_iter)):

            ctf = ctfs[k*batch_size:(k+1)*batch_size].to(device)
            rotation = rotations[k*batch_size:(k+1)*batch_size].to(device)
            tran = trans[k*batch_size:(k+1)*batch_size].to(device)
            meta = metas[k*batch_size:(k+1)*batch_size].to(device)

            par_gen = model.generate(rotation=rotation, trans=tran, ctf_para=ctf, meta=meta)
            par_gen = par_gen.detach().cpu().squeeze().numpy().astype(np.float32) / data_scale

            mrc.data[k*batch_size:(k+1)*batch_size] = par_gen

if __name__ == '__main__':
    main()
