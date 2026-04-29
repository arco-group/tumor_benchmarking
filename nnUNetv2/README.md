# nnU-Net

spiegare che abbiamo usato l'implementazione ufficiale riportata a https://github.com/mic-dkfz/nnunet e guardare il readme principale per vedere la struttura del directory del dataset:

```
pip install nnunetv2
```

```
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

```
nnUNetv2_train DATASET_ID CONFIGURATION all
```

```
nnUNetv2_predict -i INPUT_TEST_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION
````

