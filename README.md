
# Benchmarks of pre-trained models on common continual learning datasets

The script allows the evaluation of pre-trained models on common continual learning datasets. 

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

