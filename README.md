# Benchmarks of pre-trained models on common continual learning datasets

The script allows the evaluation of pre-trained models on common continual learning datasets. 
As the models used are trained on ImageNet1K, some datasets may not have a 1:1 mapping with this dataset; in this case, a mask between datasets is applied (i.e. CIFAR-10 -- ImageNet1K).
Tiny-ImageNet and ImageNet-R do not need a mapping, as they are a subset of ImageNet1k.

Available models:
-

- variants of ResNet;
- variants of ViT.

Each model is loaded from PyTorch's hub along with its default pre-trained weights. 

Available datasets
-

- CIFAR10;
- CIFAR100;
- Tiny-ImageNet;
- ImageNet-R.

The preprocessing applied on each dataset is the model's default inference transform recipe.

Usage 
-

To use the script, run:

    python main.py --model={model_name} --dataset={dataset_name}

To see the list of available models and datasets names, along with optional arguments, simply run:

    python main.py -h

To add new datasets, create a new class within `datasets/eval_datasets.py` and load it in `utils/datasets.py`.

The mappings between ImageNet1k and the target datasets can be found in the `*.xlsx` files under `mappings/`. 

TODO
-

Create the `*.pkl` with the mappings of the excel files.
