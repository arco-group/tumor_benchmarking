# Swin UNETR

This directory contains the code for training and performing inference with Swin UNETR for semantic segmentation tasks.

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

To train the model, use the `train.py` script. Below is an example bash command with the required arguments:
```bash
srun --export=ALL python -u train.py \
     --data_dir tumor_benchmarking/nnUNetv2/Data/nnUNet_raw \
     --tumor tumor \
    --img_size 256 \
     -batch_size batch_size \
     -num_epoch num_epoch \
     -num_classes num_classes
```

## Inference

To run inference on the traine model, use the `inference.py` script. Below is an example bash command:

```bash
srun --export=ALL python -u inference.py \
     --data_dir tumor_benchmarking/nnUNetv2/Data/nnUNet_raw \
     --test_masks_dir tumor_benchmarking/nnUNetv2/Data/LabelsTs \
     --tumor tumor \
     --img_size 256 \
     -checkpoint_path checkpoint_path \
     -num_classes num_classes
```
