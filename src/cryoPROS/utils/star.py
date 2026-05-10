import numpy as np
import pandas as pd
import logging
import starfile
from pathlib import Path

logger = logging.getLogger(__name__)

def R_from_relion(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> np.ndarray:
    a *= np.pi / 180.0
    b *= np.pi / 180.0
    y *= np.pi / 180.0
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    Ra = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Rb = np.array([[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]])
    Ry = np.array(([cy, -sy, 0], [sy, cy, 0], [0, 0, 1]))
    R = np.dot(np.dot(Ry, Rb), Ra)
    R[0, 1] *= -1
    R[1, 0] *= -1
    R[1, 2] *= -1
    R[2, 1] *= -1
    return R

def read_para_from_starfile(fpath : Path, Apix : float, D : int):
    '''
    Parse rotations, tranlations and CTF parameters from RELION .star file
    '''
    assert fpath.suffix == '.star'
    star = starfile.read(fpath)

    if len(star) == 1 and (0 in star or '' in star or 'images' in star):
        particles = list(star.values())[0]
        logger.info('RELION <3.1 star format')

    elif len(star) == 2 and ('optics' in star and 'particles' in star):
        optics = star['optics']
        particles = star['particles']
        Apix = optics['rlnImagePixelSize'][0]
        D = optics['rlnImageSize'][0]
        particles = pd.merge(optics, particles, left_on = 'rlnOpticsGroup', right_on = 'rlnOpticsGroup', validate = 'one_to_many')
        logger.info(f'RELION >=3.1 star format, reading pixel size: {Apix} and box size: {D}')

    else:
        raise RuntimeError('Invalid particle star')

    N = len(particles)
    logger.info(f'{N} particles found')

    # Parse rotations
    euler = np.zeros((N, 3))
    euler[:, 0] = particles['rlnAngleRot']
    euler[:, 1] = particles['rlnAngleTilt']
    euler[:, 2] = particles['rlnAnglePsi']
    logger.info('Euler angles (Rot, Tilt, Psi):')
    logger.info(euler[0])

    rots = np.asarray([R_from_relion(*x) for x in euler])
    logger.info('Converting to rotation matrix:')
    logger.info(rots[0])

    # Parse translations
    trans = np.zeros((N, 2))
    trans[:, 0] = particles['rlnOriginX'] if 'rlnOriginX' in particles else particles['rlnOriginXAngst'] / Apix
    trans[:, 1] = particles['rlnOriginY'] if 'rlnOriginY' in particles else particles['rlnOriginYAngst'] / Apix
    logger.info('Translations (pixels):')
    logger.info(trans[0])
    trans /= D

    # Parse CTF parameters
    ctfs = np.zeros((N, 9), dtype = np.float32)
    ctfs[:, 0] = D
    ctfs[:, 1] = Apix
    ctfs[:, 2] = particles['rlnDefocusU']
    ctfs[:, 3] = particles['rlnDefocusV']
    ctfs[:, 4] = particles['rlnDefocusAngle']
    ctfs[:, 5] = particles['rlnVoltage']
    ctfs[:, 6] = particles['rlnSphericalAberration']
    ctfs[:, 7] = particles['rlnAmplitudeContrast']
    ctfs[:, 8] = particles['rlnPhaseShift']
    logger.info('CTF parameters:')
    logger.info(ctfs[0])

    return rots, trans, ctfs
