import numpy as np
import pandas as pd
import mrcfile
import starfile
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

def _check_keys(table, keys, label):
    for key in keys:
        if key not in table:
            raise ValueError(f'Key {key} missed in {label}')

def _read_star_particles(param_path):
    '''Read a RELION 3.0/3.1+ STAR file and return a particle table.
    '''
    star = starfile.read(param_path, always_dict = True)

    # RELION < 3.1: single particle table.
    if len(star) == 1 and (0 in star or '' in star or 'images' in star):
        particles = star[0] if 0 in star else star[''] if '' in star else star['images']
        _check_keys(
            particles,
            [
                'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi',
                'rlnOriginX', 'rlnOriginY',
                'rlnDefocusU', 'rlnDefocusV', 'rlnDefocusAngle',
                'rlnVoltage', 'rlnSphericalAberration',
                'rlnAmplitudeContrast', 'rlnImageName',
            ],
            'particle star file',
        )
        return particles

    # RELION >= 3.1: optics and particles tables.
    if len(star) == 2 and ('optics' in star and 'particles' in star):
        optics = star['optics']
        particles = star['particles']
        _check_keys(
            optics,
            [
                'rlnOpticsGroup', 'rlnVoltage', 'rlnImagePixelSize',
                'rlnSphericalAberration', 'rlnAmplitudeContrast',
            ],
            'block data_optics',
        )
        _check_keys(
            particles,
            [
                'rlnOpticsGroup', 'rlnAngleRot', 'rlnAngleTilt',
                'rlnAnglePsi', 'rlnOriginXAngst', 'rlnOriginYAngst',
                'rlnDefocusU', 'rlnDefocusV', 'rlnDefocusAngle',
                'rlnImageName',
            ],
            'block data_particles',
        )
        try:
            particles = pd.merge(
                optics,
                particles,
                left_on = 'rlnOpticsGroup',
                right_on = 'rlnOpticsGroup',
                validate = 'one_to_many',
            )
        except pd.errors.MergeError as exc:
            raise ValueError('There are multiple optic groups with same index. Check the star file') from exc
        return particles

    raise ValueError(f'Invalid particle star file: {str(param_path)}')

def _rotation_from_relion(rot, tilt, psi):
    '''Convert RELION Euler angles in degrees to rotation matrices.
    '''
    a = np.radians(rot)
    b = np.radians(tilt)
    y = np.radians(psi)

    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)

    rots = np.empty((len(a), 3, 3), dtype = np.float32)
    rots[:, 0, 0] = cy * cb * ca - sy * sa
    rots[:, 0, 1] = cy * cb * sa + sy * ca
    rots[:, 0, 2] = -cy * sb
    rots[:, 1, 0] = -(sy * cb * ca + cy * sa)
    rots[:, 1, 1] = -sy * cb * sa + cy * ca
    rots[:, 1, 2] = sy * sb
    rots[:, 2, 0] = sb * ca
    rots[:, 2, 1] = sb * sa
    rots[:, 2, 2] = cb
    return rots

class ParticleDataset(Dataset):
    '''PyTorch dataset for RELION particle stacks.

    STAR metadata are parsed at construction time, while particle images are
    read lazily in ``__getitem__``.
    '''

    def __init__(self, opt : dict):
        super().__init__()
        self.opt = opt
        self.data_dir = Path(opt.get('data_dir', '.'))
        self.data_scale = opt.get('data_scale')
        self.param_path = Path(opt['param_path'])
        self.apix = opt['apix']
        self.box_size = opt['box_size']
        self.cached_mrc_handles = {}

        if not self.data_dir.is_dir():
            raise ValueError(f'`data_dir` should be a directory: {str(self.data_dir)}')

        if self.param_path.suffix != '.star':
            raise ValueError(f'Only RELION star files are supported: {str(self.param_path)}')

        self.particles = _read_star_particles(self.param_path)
        self._parse_image_names()
        rotations, trans, ctfs = self._parse_particle_parameters()

        rotations = rotations.transpose(0, 2, 1) # for cryoDRGN
        r = R.from_matrix(rotations).as_quat()
        metas = np.concatenate([r, trans], axis = 1)

        self.rotations = torch.from_numpy(rotations).float()
        self.trans = torch.from_numpy(trans).float()
        self.ctfs = torch.from_numpy(ctfs).float()
        self.metas = torch.from_numpy(metas).float()

        self.n = len(self.rotations)
        if opt['n_max'] is not None and opt['n_max'] < self.n:
            self.n = opt['n_max']

    def _parse_image_names(self):
        '''Parse ``rlnImageName`` into one-based slice indices and MRC paths.
        '''
        split_data = self.particles['rlnImageName'].astype(str).str.split('@', n = 1, expand = True)
        if split_data.shape[1] != 2:
            raise ValueError('Invalid rlnImageName format; expected "index@stack_path"')

        self.i_slcs = split_data[0].to_numpy(dtype = np.int64)
        self.names = split_data[1].to_numpy(dtype = np.str_)

    def _parse_particle_parameters(self):
        '''Parse rotations, translations, and CTFs using the old return layout.
        '''
        particles = self.particles
        n_particles = len(particles)

        rotations = _rotation_from_relion(
            particles['rlnAngleRot'],
            particles['rlnAngleTilt'],
            particles['rlnAnglePsi'],
        )

        pixel_size = particles['rlnImagePixelSize'][0] if 'rlnImagePixelSize' in particles else self.apix
        image_size = particles['rlnImageSize'][0] if 'rlnImageSize' in particles else self.box_size

        trans = np.empty((n_particles, 2), dtype = np.float32)
        trans[:, 0] = particles['rlnOriginX'] if 'rlnOriginX' in particles else particles['rlnOriginXAngst'] / pixel_size
        trans[:, 1] = particles['rlnOriginY'] if 'rlnOriginY' in particles else particles['rlnOriginYAngst'] / pixel_size
        trans /= image_size

        ctfs = np.empty((n_particles, 7), dtype = np.float32)
        ctfs[:, 0] = particles['rlnDefocusU']
        ctfs[:, 1] = particles['rlnDefocusV']
        ctfs[:, 2] = particles['rlnDefocusAngle']
        ctfs[:, 3] = particles['rlnVoltage']
        ctfs[:, 4] = particles['rlnSphericalAberration']
        ctfs[:, 5] = particles['rlnAmplitudeContrast']
        ctfs[:, 6] = particles['rlnPhaseShift'] if 'rlnPhaseShift' in particles else 0.0

        return rotations, trans, ctfs

    def _read_particle(self, mrc_path, i_slc):
        if mrc_path in self.cached_mrc_handles:
            mrc = self.cached_mrc_handles[mrc_path]
        else:
            mrc = mrcfile.mmap(mrc_path, mode = 'r', permissive = True)
            self.cached_mrc_handles[mrc_path] = mrc

        if mrc.data.ndim == 2:
            assert i_slc == 0, f'{str(mrc_path)} is a single image stack, while accessing {i_slc}-th image'
            return mrc.data
        return mrc.data[i_slc]

    def __getitem__(self, index):
        assert 0 <= index < self.n

        mrc_path = self.data_dir / self.names[index]
        if not mrc_path.is_file():
            raise FileNotFoundError(f'No such particle stack file: "{str(mrc_path)}"')

        img = self._read_particle(mrc_path, self.i_slcs[index] - 1)
        img = np.array(img, dtype = np.float32, copy = True)
        if self.data_scale is not None: img *= self.data_scale
        img = np.expand_dims(img, axis = 0)
        img = torch.from_numpy(img)

        ctf = self.ctfs[index]
        rotation = self.rotations[index]
        trans = self.trans[index]
        meta = self.metas[index]

        return {'img': img, 'ctf': ctf, 'rotation': rotation, 'trans': trans, 'meta': meta}

    def __len__(self):
        return self.n
