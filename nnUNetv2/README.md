# nnU-Net

This folder contains our nnU-Net v2 experiments. We used the official implementation from the nnU-Net project:
https://github.com/MIC-DKFZ/nnUNet

For the full dataset organization used across this repository, please refer to the main project README at the repository root.

## Installation

Install nnU-Net v2 with pip:

```bash
pip install nnunetv2
```

## Configure the environment

Set the nnU-Net paths before running any preprocessing or training commands:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

## Dataset structure

Each dataset must follow the nnU-Net directory conventions. In this repository, the raw data is organized so that image, label, and split files remain aligned with the codebase expectations.

The train and test split files should list relative paths in the following format:

```text
tumor_000_0000.nii.gz/0
tumor_000_0000.nii.gz/1
tumor_000_0000.nii.gz/2
tumor_000_0000.nii.gz/3
tumor_001_0000.nii.gz/0
tumor_001_0000.nii.gz/1
tumor_001_0000.nii.gz/2
tumor_001_0000.nii.gz/3
...
```

## Preprocess a dataset

Run preprocessing and integrity checks for the dataset you want to use:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

## Train a model

Train the selected configuration on all available folds:

```bash
nnUNetv2_train DATASET_ID CONFIGURATION all
```

## Optional inference

If needed, predictions can be generated with:

```bash
nnUNetv2_predict -i INPUT_TEST_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION
```

