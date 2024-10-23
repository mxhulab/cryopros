"""Parse image poses from RELION .star file"""

import logging
import numpy as np
import site
import sys
import os
site_packages_dir = site.getsitepackages()[0]
package_path = os.path.join(site_packages_dir, "cryoPROS")
sys.path.append(package_path)
import utils.utils_starfile as starfile

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

def read_pose_from_starfile(fileroot, Apix, D):
    assert fileroot.endswith(".star"), "Input file must be .star file"

    s = starfile.Starfile.load(fileroot)
    if s.relion31:  # Get image stats from data_optics table
        assert s.data_optics is not None
        assert (
            len(s.data_optics.df) == 1
        ), "Datasets with only one optics group are supported."
        Apix = float(s.data_optics.df["_rlnImagePixelSize"][0])
        D = int(s.data_optics.df["_rlnImageSize"][0])
    if D is None and "_rlnImageSize" in s.headers:
        D = int(s.df["_rlnImageSize"][0])
    assert D is not None, "Must provide image size with -D"

    N = len(s.df)
    logger.info(f"{N} particles")

    # parse rotations
    euler = np.zeros((N, 3))
    euler[:, 0] = s.df["_rlnAngleRot"]
    euler[:, 1] = s.df["_rlnAngleTilt"]
    euler[:, 2] = s.df["_rlnAnglePsi"]
    logger.info("Euler angles (Rot, Tilt, Psi):")
    logger.info(euler[0])
    logger.info("Converting to rotation matrix:")
    rot = np.asarray([R_from_relion(*x) for x in euler])

    logger.info(rot[0])

    # parse translations
    trans = np.zeros((N, 2))
    if "_rlnOriginX" in s.headers and "_rlnOriginY" in s.headers:
        # translations in pixels
        trans[:, 0] = s.df["_rlnOriginX"]
        trans[:, 1] = s.df["_rlnOriginY"]
    elif "_rlnOriginXAngst" in s.headers and "_rlnOriginYAngst" in s.headers:
        # translation in Angstroms (Relion 3.1)
        assert (
            Apix is not None
        ), "Must provide --Apix argument to convert _rlnOriginXAngst and _rlnOriginYAngst translation units"
        trans[:, 0] = s.df["_rlnOriginXAngst"]
        trans[:, 1] = s.df["_rlnOriginYAngst"]
        trans /= Apix
    else:
        logger.warning(
            "Warning: Neither _rlnOriginX/Y nor _rlnOriginX/YAngst found. Defaulting to 0s."
        )

    logger.info("Translations (pixels):")
    logger.info(trans[0])

    # convert translations from pixels to fraction
    trans /= D

    pose = (rot, trans)

    return pose
