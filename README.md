# Benchmarks of pre-trained models on common continual learning datasets

The script allows the evaluation of pre-trained models on common continual learning datasets. 
As the models used are trained on ImageNet1K, some datasets may not have a 1:1 relationship with this dataset; in this case, a mask between the source and target dataset is applied (i.e. CIFAR-10 -- ImageNet1K).
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

The preprocessing applied on each dataset is the model's default one.

Usage 
-

To use the script, run:

    python main.py --model={model_name} --dataset={dataset_name}

To see the list of available models and datasets names, along with optional arguments, simply run:

    python main.py -h
