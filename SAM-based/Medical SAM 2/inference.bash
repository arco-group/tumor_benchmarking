#!/usr/bin/env bash

# Activate venv
cd /SAM-based/sam_venv|| exit
source bin/activate

# Run training
cd /mimer/NOBACKUP/groups/naiss2023-6-336/emulero/medsam2/automatic-medsam2 || exit
srun --export=ALL python -u train_3d.py \
    -net sam2 \
    -exp_name Brain_new \
    -model Brain_new \
    -dataset multi_dataset \
    -tumor brain \
    -sam_ckpt ./checkpoints/sam2_hiera_small.pt \
    -sam_config sam2_hiera_s \
    -image_size 1024 \
    -prompt bbox \
    -data_path /mimer/NOBACKUP/groups/naiss2023-6-336/emulero/nnUNetv2/Data/nnUNet_raw \
    -pretrain ./checkpoints/MedSAM2_pretrain.pth \

# Deactivate venv
deactivate