import numpy as np
import mrcfile
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from ..utils import read_para_from_starfile

class Dataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.data_path = opt['data_path']
        self.data_scale = opt['data_scale']
        self.param_path = opt['param_path']

        with mrcfile.mmap(self.data_path, mode = 'r', permissive = True) as mrc:
            self.data = mrc.data

        if self.param_path.endswith('.star'):
            rotations, trans, ctfs = read_para_from_starfile(self.param_path, opt['Apix'], opt['box_size'])

        ctfs = ctfs[:, 2:]
        rotations = rotations.transpose(0, 2, 1) # cryoDRGN
        r = R.from_matrix(rotations)
        r = r.as_quat()
        metas = np.concatenate([r, trans], axis = 1)

        self.rotations = torch.from_numpy(rotations).float()
        self.trans = torch.from_numpy(trans).float()
        self.ctfs = torch.from_numpy(ctfs).float()
        self.metas = torch.from_numpy(metas).float()

        assert self.data.shape[0] >= self.rotations.shape[0]
        self.n = len(self.rotations)

        if opt['n_max'] is not None and opt['n_max'] < self.n:
            self.n = opt['n_max']

    def __getitem__(self, index):
        assert 0 <= index < self.n

        img = np.array(self.data[index], dtype = np.float32, copy = True)
        img *= self.data_scale
        img = np.expand_dims(img, axis = 0)
        img = torch.from_numpy(img)

        ctf = self.ctfs[index]
        rotation = self.rotations[index]
        trans = self.trans[index]
        meta = self.metas[index]

        return {'img': img, 'ctf': ctf, 'rotation': rotation, 'trans': trans, 'meta': meta}

    def __len__(self):
        return self.n
