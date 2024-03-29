import numpy as np
from argparse import Namespace

import matplotlib.pyplot as plt

import torchvision as tv
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from datasets import eval_datasets
from utils.conf import base_path

def get_dataset(args):
    """
    Get the test_set for the evaluation experiment
    :return: dataset
    """
    dataset = args.dataset
    
    transform_weights = models.get_model_weights(args.model)
    default_weights = transform_weights.DEFAULT

    #New preprocessings can be added here
    preprocess = default_weights.transforms()

    print(preprocess)
    
    if dataset == 'cifar10':
        test_dataset = eval_datasets.CIFAR10(root=base_path(),train=False,transform=preprocess,download=True)
    elif dataset == 'cifar100':
        test_dataset = eval_datasets.CIFAR100(root=base_path(),train=False,transform=preprocess,download=True)
    elif dataset == 'tinyimagenet':
        test_dataset = eval_datasets.TinyImagenet(root=base_path(),transform=preprocess,download=True)
    elif dataset == 'tinyimagenet-hd':
        test_dataset = eval_datasets.TinyImagenetHD(root=base_path(),transform=preprocess,download=True)
    elif dataset == 'imagenet-r':
        test_dataset = eval_datasets.ImagenetR(root=base_path(),transform=preprocess,download=True)
    else:
        raise NotImplementedError('Unknown dataset: ' + dataset)
        
    return test_dataset

def get_dataloader(dataset):
    """
    Creates the dataloader for the desired dataset
    :return: dataloader
    """
    return DataLoader(dataset,batch_size=32,shuffle=False,drop_last=False)

def dl_mask_eligible(dataset):
    """
    A dataloader mask can only be applied on datasets that do not have a 1:1 relationship with the source target (i.e. CIFAR10-ImageNet)
    Assert that the dataset is eligible for a dataloader mask.
    """
    DS_ELIGIBLE = ['cifar10','cifar100']
    try:
        assert dataset.DS_NAME in DS_ELIGIBLE
    except AssertionError:
        print("A dataloader mask cannot be applied on this dataset!")
    else:
        return True
        
def imshow(img_batch):
    """
    Display image
    """
    img = tv.utils.make_grid(img_batch)
    img = img / 2 + 0.5
    npimg = img.numpy()   
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    plt.show()
    
    #Use this function by calling imshow(tv.utils.make_grid(image))
