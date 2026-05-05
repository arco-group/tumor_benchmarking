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

**Important**: You must modify this dictionary to match your own dataset folder names and tumor types. Each key should correspond to the tumor type you use as the `-tumor` argument, and each value should be the exact folder name in your `nnUNet_raw` directory.

## Training

From the project root, you can launch a single-GPU training with:

```bash
srun --export=ALL python -u train_single_gpu.py \
    -data_dir /nnUNetv2/Data/nnUNet_raw \
    -tumor tumor \
    -num_epochs num_epochs \
    -batch_size batch_size \
    -use_wandb True \
```

For multi-GPU training, launch DDP with `torchrun`:

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
    -num_epochs num_epochs \
    -batch_size batch_size_per_gpu \
    -use_wandb True
```

`NUM_GPUS` should match the number of available GPUs. The `-batch_size` argument is the batch size per GPU.

To resume a previous run, add:

```bash
-resume /path/to/checkponts/medsam_model_latest.pth
```

## Inference

From the project root, you can launch inference with:

```bash
srun --export=ALL python -u inference.py \
    -data_dir /nnUNetv2/Data/nnUNet_raw \
    -test_masks_dir /nnUNetv2/Data/LabelsTs \
    -tumor tumor \
    -checkpoint checkpoint_path \
    -rescale_bbox None \
    -shift_percent None \
    -zero-shot False
```
For zero-shot inference, set `-sam_ckpt sam_vit_b_01ec64.pth` or change path on default values in the parser of inference.py.

The `-rescale_bbox` and `-shift_percent` arguments can be used to tighten, expand, or shift bounding boxes during inference.
