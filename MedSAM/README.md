# MedSAM

This repository contains the code used to train and evaluate MedSAM on custom medical imaging datasets.

## Installation

Install MedSAM and download the required weights by following the instructions in the [official repository](https://github.com/bowang-lab/MedSAM/tree/main).

## Using a Custom Dataset

In the `dataset.py` file, the `MultiDataset` class contains a dictionary called `tumor_path_labels` that maps tumor types to their corresponding dataset folder names in the nnUNet_raw directory. This dictionary looks like:

```python
tumor_path_labels = {
    'lung': 'Dataset002_Lung1',
    'breast': 'Dataset008_ISPY1',
    'liver': 'Dataset009_Liver2',
    'kidney': 'Dataset012_Kidney',
    'brain': 'Dataset004_BraTS'
}
```

**Important**: You must modify this dictionary to match your own dataset folder names and tumor types. Each key should correspond to the tumor type you use as the `--tumor` argument, and each value should be the exact folder name in your `nnUNet_raw` directory.

## Training

Run training from the `MedSAM` directory.

Single GPU:

```bash
python -u train_single_gpu.py \
    --data_dir /nnUNetv2/Data/nnUNet_raw \
    --tumor tumor \
    -num_epochs 100 \
    -batch_size 2 \
    -use_wandb False
```

Multi-GPU with DDP:

```bash
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

torchrun \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    train_multi_gpu.py \
    --data_dir /nnUNetv2/Data/nnUNet_raw \
    --tumor tumor \
    -num_epochs 100 \
    -batch_size 2 \
    -use_wandb False
```

`-batch_size` is the batch size per GPU, so the effective batch size is `batch_size x NUM_GPUS`.

Useful optional arguments:

```bash
-checkpoint sam_vit_b_01ec64.pth
-resume /path/to/MedSAM_tumor-YYYYMMDD-HHMMSS/medsam_model_latest.pth
```

## Inference

Run inference from the `MedSAM` directory.

Fine-tuned MedSAM model:

```bash
python -u inference.py \
    --data_dir /nnUNetv2/Data/nnUNet_raw \
    --test_masks_dir /nnUNetv2/Data/LabelsTs \
    --tumor tumor \
    -checkpoint /path/to/MedSAM_tumor-YYYYMMDD-HHMMSS/medsam_model_best.pth \
    -zero_shot False
```

Zero-shot SAM baseline:

```bash
python -u inference.py \
    --data_dir /nnUNetv2/Data/nnUNet_raw \
    --test_masks_dir /nnUNetv2/Data/LabelsTs \
    --tumor tumor \
    -checkpoint sam_vit_b_01ec64.pth \
    -zero_shot True
```

Optional bounding-box controls:

```bash
-rescale_bbox 1.05
-shift_percent 0.10
```

Leave them unset to keep the default bounding boxes. 

## Notes

`-use_wandb` is disabled by default. If you enable it, make sure your Weights & Biases credentials are configured first.

