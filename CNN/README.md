# CNN Models

This directory contains tools for training and performing inference with CNN models for semantic segmentation tasks.

## Overview

You can train DeepLabV3 and other torchvision semantic segmentation models (see [PyTorch Vision Models](https://docs.pytorch.org/vision/0.9/models.html#semantic-segmentation)) as well as U-Net from a single training file. Inference can be performed using a separate inference file.

## Dataset Configuration

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

To train a model, use the `train.py` script. Below is an example bash command with the required arguments:

```bash
source venv/bin/activate

srun --export=ALL python -u train.py \
     --data_dir tumor_benchmarking/nnUNetv2/Data/nnUNet_raw \
     --tumor tumor \
     --model model_name \
     -batch_size batch_size \
     -num_epoch num_epoch \
     -num_classes num_classes

deactivate
```

### Training Arguments

- `--data_dir`: Path to the directory containing the training data
- `--tumor`: Type of tumor to train on
- `--model`: Model architecture to use (e.g., DeepLabV3, U-Net)
- `-batch_size`: Batch size for training
- `-num_epoch`: Number of training epochs
- `-num_classes`: Number of classes for segmentation

## Inference

To run inference on trained models, use the `inference.py` script. Below is an example bash command:

```bash
source venv/bin/activate

srun --export=ALL python -u inference.py \
     --data_dir tumor_benchmarking/nnUNetv2/Data/nnUNet_raw \
     --test_masks_dir tumor_benchmarking/nnUNetv2/Data/LabelsTs \
     --tumor tumor \
     --model model_name \
     -checkpoint_path checkpoint_path \
     -batch_size batch_size \
     -num_classes num_classes

deactivate
```

### Inference Arguments

- `--data_dir`: Path to the directory containing the test data
– `--test_masks_dir`: Path to the directory containing the test data
- `--tumor`: Type of tumor to perform inference on
- `--model`: Model architecture used for training
- `-checkpoint_path`: Path to the trained model checkpoint
- `-batch_size`: Batch size for inference
- `-num_classes`: Number of classes for segmentation