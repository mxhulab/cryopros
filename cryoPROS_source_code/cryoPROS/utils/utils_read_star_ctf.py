"""Parse CTF parameters from a RELION .star file"""

import argparse
import os
import pickle
import logging
import numpy as np
import site
import sys
site_packages_dir = site.getsitepackages()[0]
package_path = os.path.join(site_packages_dir, "cryoPROS")
sys.path.append(package_path)
import utils.utils_ctf as ctf
import utils.utils_starfile as starfile

logger = logging.getLogger(__name__)

HEADERS = [
    "_rlnDefocusU",
    "_rlnDefocusV",
    "_rlnDefocusAngle",
    "_rlnVoltage",
    "_rlnSphericalAberration",
    "_rlnAmplitudeContrast",
    "_rlnPhaseShift",
]

def read_ctf_from_starfile(fileroot, Apix, D):
    assert fileroot.endswith(".star"), "Input file must be .star file"

    s = starfile.Starfile.load(fileroot)
    N = len(s.df)
    logger.info(f"{N} particles")

    overrides = {}
    if s.relion31:
        assert s.data_optics is not None
        df = s.data_optics.df
        assert len(df) == 1, "Only one optics group supported"
        D = int(df["_rlnImageSize"][0])
        Apix = float(df["_rlnImagePixelSize"][0])
        overrides[HEADERS[3]] = float(df[HEADERS[3]][0])
        overrides[HEADERS[4]] = float(df[HEADERS[4]][0])
        overrides[HEADERS[5]] = float(df[HEADERS[5]][0])
    else:
        assert D is not None, "Must provide image size with -D"
        assert Apix is not None, "Must provide pixel size with --Apix"

    ctf_params = np.zeros((N, 9))
    ctf_params[:, 0] = D
    ctf_params[:, 1] = Apix
    for i, header in enumerate(
        [
            "_rlnDefocusU",
            "_rlnDefocusV",
            "_rlnDefocusAngle",
            "_rlnVoltage",
            "_rlnSphericalAberration",
            "_rlnAmplitudeContrast",
            "_rlnPhaseShift",
        ]
    ):
        ctf_params[:, i + 2] = (
            s.df[header] if header not in overrides else overrides[header]
        )

    return ctf_params.astype(np.float32)
