# cryoPROS

# Tutorial

## Step 1 (Optional, Reconstruct micelle for membrane protein) 

First, generate a volume mask with a user-chosen threshold value:
```shell
python cryopros_gen_mask.py \
--volume_path latent_volume.mrc \             # Path to latent volume
--result_path ./mask.mrc \                    # Path to volume mask
--threshold 0.1 \                             # Threshold value, chosen by the user
```

Second, reconstruct volume with micelle:
```shell
python cryopros_recondismic.py \
--opt ./options/train_mp.json \              # Fixed
--gpu_ids 0 1 2 3 \                          # GPU id
--task_name task_name \                      # Task name
--box_size 256 \                             # Box size
--Apix 1.0 \                                 # Pixel size
--init_volume_path latent_volume.mrc \       # Path to latent volume
--volume_scale 10 \                          # Volume scale
--mask_path ./mask.npy \                     # Path to volume mask
--data_path raw_particles.mrcs \             # Path to raw particle
--param_path param.star \                    # Path to starfile
--invert \                                   # Invert projection
--dataloader_batch_size 24 \                 # Batch size
```
The reconstructed density can be found in `./reconstruct/task_name/results/`.


## Step 2
Model training:
```shell
python cryopros_train.py \
--opt ./options/train.json \                    # Fixed
--gpu_ids 0 1 2 3 \                             # GPU id
--task_name task_name \                         # Task name
--box_size 256 \                                # Box size
--Apix 1.0 \                                    # Pixel size
--volume_scale 50 \                             # Latent volume scale: 50 or 100
--init_volume_path latent_volume.mrc \          # Path to latent volume
--data_path raw_particles.mrcs \                # Path to raw particle
--param_path param.star \                       # Path to starfile
--invert \                                      # Invert projection
--dataloader_batch_size 8 \                     # Batch size
```
The model weights can be found in `./generate/task_name/models/`.

## Step 3

particle generation:
```shell
python cryopros_generate.py \
--model_path model.pth \               # Path to pre-trained model in step 2
--param_path param.star \              # Path to starfile
--output_path ./generate/ \            # Path to generated particles
--gen_name generated_particles \       # Name of generated particles
--batch_size 50 \                      # Generation batch size
--box_size 256 \                       # Box size
--Apix 1.0 \                           # Pixel size
--invert \                             # Invert projection
```

## Step 4
Executing "Non-uniform Refinement" on software CryoSPARC:
```
Particle stacks = raw particle stack + generated particle stack
Initial volume = latent volume used by cryoPROS in this round
Initial lowpass resolution (A) = 30
Set Symmetry according to sample properties, and select default values for other parameters.
```

## Step 5
Executing "Homogeneous Reconstruction Only" on software CryoSPARC:
```
Particle stacks = raw particle stack (with corrected orientations in step 4)
Set Symmetry according to sample properties, and select default values for other parameters.
```

## Step 6
Update latent volume as the reconstructed map in step 5, and re-prepare the ctf.pkl and pose.pkl file from star file obteined in step 5, then retraining CVAE network.
