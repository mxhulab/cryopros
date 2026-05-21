# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

- The Python package is under `cryoPROS_source_code/cryoPROS/`; packaging metadata is in `cryoPROS_source_code/pyproject.toml`.
- The top-level `README.md` is a short usage tutorial; `cryoPROS_source_code/README.md` contains the detailed installation/tutorial/options documentation.
- Console scripts are defined in `pyproject.toml`:
  - `cryopros-train` -> `cryoPROS.cryopros_train:main`
  - `cryopros-generate` -> `cryoPROS.cryopros_generate:main`
  - `cryopros-gen-mask` -> `cryoPROS.cryopros_gen_mask:main`
  - `cryopros-recondismic` -> `cryoPROS.cryopros_recondismic:main`

## Development commands

Run commands from the repository root unless noted.

```bash
# Create the environment shown in the README
conda create -n cryopros python==3.10
conda activate cryopros

# Install in editable mode for development
python -m pip install -e ./cryoPROS_source_code

# Build the package distribution; requires the `build` package to be installed
cd cryoPROS_source_code && python -m build

# Verify CLI installation / inspect arguments
cryopros-generate -h
cryopros-train -h
cryopros-recondismic -h
cryopros-gen-mask -h

# Basic syntax validation; no repository test suite is currently present
python -m compileall cryoPROS_source_code/cryoPROS
```

No linter, formatter, or test runner is configured in `pyproject.toml`, and no test files/directories are currently present. There is therefore no repository-specific single-test command yet.

## Runtime workflows

The package is designed for cryo-EM preferred-orientation correction using generated auxiliary particles:

1. Optionally create a mask from an MRC volume with `cryopros-gen-mask`.
2. Optionally reconstruct micelle/nanodisc density with `cryopros-recondismic`.
3. Train the generative conditional VAE with `cryopros-train`.
4. Generate synthetic auxiliary particles and a corresponding STAR file with `cryopros-generate`.
5. Co-refine raw and generated particle stacks in external cryo-EM software such as CryoSPARC/Relion, then iterate with updated poses/volumes.

Training writes outputs relative to the process working directory:

- `cryopros-train` uses `options/train.json` and writes under `generate/<task_name>/`.
- `cryopros-recondismic` uses `options/train_mp.json` and writes under `reconstruct/<task_name>/`.
- Saved options go under `<task>/options/`, logs under `<task>/train.log`, models/volumes under `<task>/models/`, and generated training preview images under `<task>/images/` when applicable.

The README notes that a very low KL loss during `cryopros-train` (for example around `1e-9`) usually indicates posterior collapse and the training run should be restarted.

## High-level architecture

### Option parsing and GPU setup

`cryoPROS/utils/utils_option.py` is the shared option layer for training CLIs. It reads JSON option files that contain `//` comments, overlays CLI arguments, creates derived output paths, and sets `CUDA_VISIBLE_DEVICES` from `--gpu_ids` before model construction.

### Data and metadata pipeline

- Particle images/volumes are read with `mrcfile` from `.mrcs` and `.mrc` files.
- STAR metadata is parsed by `cryoPROS/utils/utils_starfile.py` and converted by:
  - `utils_read_star_pose.py` for RELION Euler angles/translations.
  - `utils_read_star_ctf.py` for CTF parameters.
- RELION 3.1 `data_optics` STAR files are supported only when there is one optics group.
- Datasets drop the first two CTF columns (`D`, `Apix`) before passing CTF tensors to models; model CTF functions expect the remaining seven parameters.
- Rotations are transposed for cryoDRGN convention before training/generation.
- `data/dataset.py` is used by `cryopros-train`; it returns scaled image tensors plus CTF, rotation, translation, and a `meta` vector made from quaternion + translation.
- `data/dataset_mp.py` is used by `cryopros-recondismic`; it returns image tensors plus CTF, rotation, and translation without the VAE `meta` vector.

### Generative module

`cryopros-train` builds `models/model_hvae.py::HVAEModel`, which wraps `models/network_hvae.py::HVAE` in `torch.nn.DataParallel`, optimizes it with Adam/MultiStepLR, and saves plain PyTorch `state_dict` `.pth` files.

`HVAE` combines:

- A fixed or optionally trainable initial 3D latent volume projected through a differentiable 3D Radon transform, 2D translation, and CTF convolution.
- A hierarchical VAE encoder/decoder conditioned on the degraded projection and pose metadata.
- Generation via `HVAE.generate()`, which uses the learned decoder without an input particle image.

`cryopros-generate` loads a saved `HVAE` state dict, creates a uniform-pose STAR file through `utils/utils_pose.py::generate_uniform_pose`, reads pose/CTF metadata from that generated STAR file, and writes generated particles to `<output_path>/<gen_name>.mrcs`.

### Micelle/nanodisc reconstruction module

`cryopros-recondismic` builds `models/model_mp.py::ReconModel`, which wraps `models/network_mp.py::Reconstructor`. The reconstructor projects a volume through the same differentiable projection/CTF path and optimizes MSE against raw particles. Its reconstructed volume combines the fixed/scaled initial volume with a learned micelle component outside the mask.

## Import/path convention

Many modules append the installed `site-packages/cryoPROS` directory to `sys.path` and then import modules as top-level `data`, `models`, and `utils`. Prefer installing the package in editable mode before running CLIs or source scripts; direct script execution from an uninstalled checkout can fail unless `PYTHONPATH` points at `cryoPROS_source_code/cryoPROS`.
