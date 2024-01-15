from argparse import Namespace
from datasets import eval_datasets

import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataset(args: Namespace):
    """
    Get the test_set for the evaluation experiment
    :return: dataset
    """
    dataset = args.dataset

    transform_weights = models.get_model_weights(args.model)
    default_weights = transform_weights.DEFAULT

    #New preprocessings can be added here
    preprocess = default_weights.transforms()
    
    if dataset == 'cifar10':
        test_dataset = eval_datasets.CIFAR10(root='../data/',train=False,transform=preprocess,download=True)
    elif dataset == 'cifar100':
        test_dataset = eval_datasets.CIFAR100(root='../data/',train=False,transform=preprocess,download=True)
    elif dataset == 'tinyimagenet':
        test_dataset = eval_datasets.TinyImagenet(root='../data/',transform=preprocess)
    elif dataset == 'tinyimagenet-hd':
        test_dataset = eval_datasets.TinyImagenetHD(root='../data/',transform=preprocess,download=True)
    elif dataset == 'tinyimagenet-r':
        test_dataset = eval_datasets.TinyImagenetR(root='../data/',transform=preprocess,download=True)
    else:
        raise NotImplementedError('Unknown dataset: ' + dataset)
        
    return dataset

def get_dataloader(dataset):
    """
    Creates the dataloader for the desired dataset
    :return: dataloader
    """
    return DataLoader(dataset,batch_size=args.batch_size,shuffle=False,drop_last=False)
