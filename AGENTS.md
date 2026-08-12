# CryoPROS Development Guide

## Layout and commands

This repository uses a standard `src` layout. Packaging metadata is in the root
`pyproject.toml`, Python sources are under `src/cryoPROS/`, and the module
contract consumed by CoCo is `module.yaml`.

Python `>=3.9` is required. Run commands from the repository root:

```bash
python -m pip install -e .
python -m pip install build
python -m build
python -m compileall src/cryoPROS
cryopros-genmask -h
cryopros-recondismic -h
cryopros-train -h
cryopros-generate -h
```

No linter, formatter, or automated test runner is configured. Use package
build, compile validation, CLI help, and focused runs on representative MRC/STAR
data. Do not add generated `build/`, `dist/`, egg metadata, model output, or
Python caches.

## Runtime workflow

CryoPROS corrects preferred-orientation problems with generated auxiliary
particles. The main flow is mask generation, optional micelle/nanodisc
reconstruction, conditional VAE training, auxiliary-particle generation, and
external co-refinement with updated poses/volumes.

Console scripts map to `cryoPROS.genmask`, `cryoPROS.recondismic`,
`cryoPROS.train`, and `cryoPROS.generate`. Keep these entry points and the
commands declared by `module.yaml` stable because CoCo generates stage scripts
against them. CryoSPARC particle export is owned by CoCo core and must not be
declared or implemented as a CryoPROS module command or resource.

Outputs are relative to the job/process working directory. Preserve stable
option snapshots, log names, model/volume paths, preview images, generated MRC
stacks, and STAR outputs because CoCo progress and result parsers consume them.
Package job logging must use the existing explicit file-handler pattern rather
than Python's implicit stderr logger.

## Data and numerical conventions

- STAR metadata and MRC/MRCS data form the public data boundary. Preserve
  RELION optics-table handling, CTF column ordering, pixel-size units, and
  translation signs.
- Datasets pass CTF, rotation, translation, and quaternion-derived pose metadata
  to the networks. Do not reorder tensor fields or silently change scaling.
- The projection, translation, and CTF paths are shared by training, generation,
  and micelle reconstruction. Changes require representative numerical checks
  across all affected commands.
- Rotations are adapted to the conventions expected by the existing
  cryo-EM/ML pipeline. Add round-trip or fixed-pose regressions before changing
  transposes, Euler conversion, quaternion order, handedness, or active/passive
  interpretation.
- Preserve state-dict compatibility where practical. If a model architecture or
  checkpoint schema must change, make the incompatibility explicit and update
  loading errors and user documentation.

The CoCo superproject owns job mappings outside this checkout. Do not add a
second mapping fragment here. Coordinate any command, artifact, capability, or
job-type contract change with `module.yaml` and the parent repository.
