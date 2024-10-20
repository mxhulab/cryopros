import torch
import numpy as np
import torch.utils.data as data
import mrcfile
import pickle
import site
import sys
import os
site_packages_dir = site.getsitepackages()[0]
package_path = os.path.join(site_packages_dir, "cryoPROS")
sys.path.append(package_path)
from scipy.spatial.transform import Rotation as R
from utils.utils_read_star_ctf import read_ctf_from_starfile
from utils.utils_read_star_pose import read_pose_from_starfile


class Dataset(data.Dataset):
    def __init__(self, opt):
        super(Dataset, self).__init__()
        self.opt = opt
        self.data_path = opt['data_path']
        self.data_scale = opt['data_scale']
        self.param_path = opt['param_path']
        
        with mrcfile.mmap(self.data_path, mode='r', permissive=True) as mrc:
            self.data = mrc.data

        if self.param_path.endswith(".star"):
            pose = read_pose_from_starfile(self.param_path, opt['Apix'], opt['box_size'])
            ctfs = read_ctf_from_starfile(self.param_path, opt['Apix'], opt['box_size'])

        rotations = pose[0]
        trans = pose[1]
        
        ctfs = ctfs[:, 2:]
        
        rotations = rotations.transpose(0, 2, 1) # cryoDRGN
        
        r = R.from_matrix(rotations)
        r = r.as_quat()
        metas = np.concatenate([r, trans], axis=1)
        
        self.rotations = torch.from_numpy(rotations).float()
        self.trans = torch.from_numpy(trans).float()
        self.ctfs = torch.from_numpy(ctfs).float()
        self.metas = torch.from_numpy(metas).float()
        
        assert self.data.shape[0] == self.rotations.shape[0]
        
        n_max = opt['n_max']
        if n_max is not None:
            self.data = self.data[:n_max]
            self.rotations = self.rotations[:n_max]
            self.trans = self.trans[:n_max]
            self.ctfs = self.ctfs[:n_max]
            self.metas = self.metas[:n_max]
            
    def __getitem__(self, index):
        img = self.data[index].copy().astype(np.float32)
        
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(np.ascontiguousarray(img)).float()
        img = self.data_scale * img
        
        ctf = self.ctfs[index]
        rotation = self.rotations[index]
        trans = self.trans[index]
        meta = self.metas[index]
            
        return {'img': img, 'ctf': ctf, 'rotation': rotation, 'trans': trans, 'meta': meta}

    def __len__(self):
        return self.data.shape[0]

