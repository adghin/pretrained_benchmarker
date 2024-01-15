"""
Benchmarks of pre-trained pytorch models on common continual learning datasets
@author: adrian.ghinea@outlook.it
"""
import pickle
import argparse
import importlib

import utils
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision as tv
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.conf import get_device
from utils.datasets import get_dataset, get_dataloader
from mappings.masks import maskSoftmax, maskDataloader

def parseArgs():
    """
    Arguments to use
    """
    parser = argparse.ArgumentParser(description="Benchmark of pre-trained models on common continual learning datasets")

    parser.add_argument("-m"      , "--model", type=str, help="Available models: resnet18, resnet34, resnet50, resnet101, resnet152, vit_b_16, vit_b_32, vit_l_16",required=True)
    parser.add_argument("-d"      , "--dataset", type=str, help="Available datasets: cifar10, cifar100, tinyimagenet, tinyimagenet-hd, tinyimagenet-r",required=True)
    parser.add_argument("-md"     , "--mask_dataloader", type=bool, help="Remove datasets' samples that doesn't have any matching with the dataset on which the model was pretrained on (FOR CIFAR-10 & CIFAR-100)", default=False)
    parser.add_argument("-gpu"    , "--gpu", type=int, help="Choose on which GPU to run the experiment",default=0)

    args = parser.parse_args()

    return args

def get_model(model):
    """
    Load the pre-trained model with default weights from torchvision.models
    """
    model_name = model
    model_weights = "DEFAULT"

    try: #PyTorch models
        model = models.get_model(model_name,weights=model_weights)
    except NameError:
        raise NameError("Unknown model: " + model_name)
    else:
        return model

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()   
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    plt.show()

    #Use this function by calling imshow(tv.utils.make_grid(image))

def evaluateModel(model,dataset,device):
    """
    Evaluate model's accuracy (correct/total).
    """
    if dataset == 'cifar10' or dataset == 'cifar100':
        ground_truth = groundTruth(dataset)                                                #mappings between CIFAR and Imagenet datasets
    else:
        ground_truth = None

    test_loader = get_dataloader(dataset)
    
    model.to(device)
    model.eval()

    correct, total = 0, 0
    
    with torch.no_grad():
        for image,label in tqdm(test_loader):
            if((args.mask_dataloader == True)):
                assert args.dataset == 'cifar10' or args.dataset == 'cifar100', "A dataloader mask can only be applied on CIFAR-10 and CIFAR-100 dataset!"

                masked_image, masked_label = maskDataloader(image,label,args.dataset)
                image, label = masked_image, masked_label
            
            image, label = image.to(device), label.to(device)

            prediction      = model(image).softmax(dim=1)
            prediction_mask = maskSoftmax(prediction,args.dataset)                              #remove unaccepted labels (i.e. labels of imagenet that doesn't have a match with any label of cifar10/100)

            preds           = [idx.argmax().item() for idx in prediction_mask]                  #get class_id from predictions
            ground_labels   = [idx.item() for idx in label]                                     #ground_labels: list of truth labels

            if ground_truth:
                correct += sum(label in ground_truth[ground_labels[i]] for i,label in enumerate(preds))
            else:
                correct += sum(label == ground_labels[i] for i,label in enumerate(preds))

            total += label.shape[0]
    
    print("\n\nCorrect: " + str(correct))
    print("\nTotal: " + str(total))
    print('\nTest Accuracy: {:.2f}%'.format(100*correct/total))

def main():
    args = parseArgs()

    dataset = get_dataset(args)
    device  = get_device(args.gpu)
    model   = get_model(args.model)

    print(dataset)
    print(type(dataset))

    evaluateModel(model,dataset,device)

main()
