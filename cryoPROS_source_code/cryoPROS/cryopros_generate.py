import numpy as np
import torch
import argparse
import os
import mrcfile
from tqdm import tqdm
import site
import sys
site_packages_dir = site.getsitepackages()[0]
package_path = os.path.join(site_packages_dir, "cryoPROS")
sys.path.append(package_path)
from scipy.spatial.transform import Rotation as R
from utils.utils_read_star_ctf import read_ctf_from_starfile
from utils.utils_read_star_pose import read_pose_from_starfile
from utils.utils_pose import generate_uniform_pose

def parse_argument():

    parser = argparse.ArgumentParser(description='Generating an auxiliary particle stack from a pre-trained conditional VAE deep neural network model.')

    parser.add_argument('--model_path' , type=str  , help='input pretrained model path'                            , required = True)
    parser.add_argument('--param_path' , type=str  , help='path of star file which contains the imaging parameters', required = True)

    parser.add_argument('--output_path', type=str  , help='output output synthesized auxiliary particle stack'     , required = True)
    parser.add_argument('--gen_name'   , type=str  , help='filename of the generated auxiliary particle stack'     , required = True)

    parser.add_argument('--box_size'   , type=int  , help='box size'                                               , required = True)
    parser.add_argument('--Apix'       , type=float, help='pixel size in Angstrom'                                 , required = True)
    parser.add_argument("--invert"     , action='store_true'          , help='invert the image sign')
    parser.add_argument('--batch_size' , type=int  , default=10       , help='batch size')
    parser.add_argument('--num_max'    , type=int  , default=100000000, help='maximum number particles to generate')
    parser.add_argument('--data_scale' , type=float, default=0.1      , help='scale factor')
    parser.add_argument('--gen_mode'   , type=int  , default=2        , help='storage model of the synthesized particles; mode 0 is int; mode 2 is float')
    parser.add_argument('--nls'        , type=int  , default=[2, 2, 4, 4], nargs='+', help='number of layers of the neural network')

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

    from models.network_hvae import HVAE
    model = HVAE(nf=64, nls=nls, z_dim=16, box_size=box_size, 
                 Apix=Apix, invert=invert)
    
    states = torch.load(model_path)

    model.load_state_dict(states, strict=True)
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
