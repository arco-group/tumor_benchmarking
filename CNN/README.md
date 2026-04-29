# CNN models

dire qua che da un solo file si puo addestrare DeepLabV3 (o altri modelli torchvision per semantic segmentation) e U-Net e da un'altro file l'inferenza

spiegare bash file e argomenti che servono

spiegare ancora che i dataset devono essere sempre nello stesso directory (nnUNet_raw) e quello sarà data_dir

```
source venv/bin/activate

srun --export=ALL python -u train.py \
     --data_dir tumor_benchmarking/nnUNetv2/Data/nnUNet_raw \
     --tumor tumor \
     --model model_name \
     -batch_size batch_size\
     -num_epoch num_epoch \
     -num_classes num_classes \

deactivate
```


```
source venv/bin/activate

srun --export=ALL python -u inference.py \
     --data_dir tumor_benchmarking/nnUNetv2/Data/nnUNet_raw \
     --tumor tumor \
     --model model_name \
     -checkpoint_path checkpoint_path \
     -batch_size batch_size\
     -num_classes num_classes \

deactivate
```