# Medical SAM 2

qua dire che si deve istallare medical sam 2 seguendo le istruzioni di https://github.com/ImprintLab/Medical-SAM2, mettere i due file train e inference (forse anche il bash file) sul directory generale e multi_dataset.py sul directory func_3d/dataset e su func_3d/dataset/__init__.py si deve aggiungere questo pezzo di codice nella riga 45

```python
    elif args.dataset == 'multi_dataset':
        multi_train_dataset = MultiDataset(args, args data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        multi_test_dataset = MultiDataset(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(multi_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(multi_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
````

poi dire che si fa il run dal bash file con i argomenti desiderati per train e inference