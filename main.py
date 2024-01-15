"""
Benchmarks of pre-trained pytorch models on common continual learning datasets.
@author: adrian.ghinea@outlook.it
@version: final release, 15/01/2024
"""
import argparse

import torch
import torchvision.models as models

from tqdm import tqdm

from utils.conf import get_device
from utils.masks import maskSoftmax, maskDataloader, groundTruth
from utils.datasets import get_dataset, get_dataloader, imshow, dl_mask_eligible

def parseArgs():
    """
    Arguments to use
    """
    parser = argparse.ArgumentParser(description="Benchmark of pre-trained models on common continual learning datasets")

    parser.add_argument("-m"      , "--model", type=str, help="Available models: resnet18, resnet34, resnet50, resnet101, resnet152, vit_b_16, vit_b_32, vit_l_16",required=True)
    parser.add_argument("-d"      , "--dataset", type=str, help="Available datasets: cifar10, cifar100, tinyimagenet, tinyimagenet-hd, imagenet-r",required=True)
    parser.add_argument("-md"     , "--mask_dataloader", type=bool, help="Remove datasets' samples that doesn't have any matching with the dataset on which the model was pretrained on (FOR CIFAR-10 & CIFAR-100)", default=False)
    parser.add_argument("-gpu"    , "--gpu", type=int, help="Choose on which GPU to run the experiment",default=0)

    args = parser.parse_args()

    return args

def get_model(model):
    """
    Load the pre-trained model with its default weights
    :return: model instance
    """
    model_name = model
    model_weights = 'DEFAULT'

    try:
        model = models.get_model(model_name,weights=model_weights)
    except NameError:
        raise NameError('Unknown model: ' + model_name)
    else:
        return model

def evaluateModel(model,dataset,device,mask_dl=False):
    """
    Evaluate model's accuracy (correct/total).
    """
    if hasattr(dataset,'DS_MASK'):                                                              #if dataset has a mask, apply it
        ground_truth = groundTruth(dataset.DS_MASK)                                             #mappings between CIFAR and Imagenet datasets
    else:
        ground_truth = None

    test_loader = get_dataloader(dataset)
    
    model.to(device)
    model.eval()

    correct, total = 0, 0
    
    with torch.no_grad():
        for image,label in tqdm(test_loader):
            if mask_dl and dl_mask_eligible(dataset):
                masked_image, masked_label = maskDataloader(image,label,dataset.DS_NAME)
                image, label = masked_image, masked_label            
                
            image, label = image.to(device), label.to(device)

            prediction      = model(image).softmax(dim=1)
            prediction_mask = maskSoftmax(prediction,dataset.DS_NAME)                           #remove unaccepted labels (i.e. labels of imagenet that doesn't have a match with any label of cifar10/100)

            preds           = [idx.argmax().item() for idx in prediction_mask]                  #get class_id from predictions
            ground_labels   = [idx.item() for idx in label]                                     #ground_labels: list of truth labels

            if ground_truth:
                correct += sum(label in ground_truth[ground_labels[i]] for i,label in enumerate(preds))
            else:
                correct += sum(label == ground_labels[i] for i,label in enumerate(preds))

            total += label.shape[0]
    
    print('\nCorrect: ' + str(correct))
    print('\nTotal: ' + str(total))
    print('\nTest Accuracy: {:.2f}%'.format(100*correct/total))

def main():
    args = parseArgs()

    dataset = get_dataset(args)
    device  = get_device(args.gpu)
    model   = get_model(args.model)

    if args.mask_dataloader:
        evaluateModel(model,dataset,device,mask_dl=True)
    else:
        evaluateModel(model,dataset,device)

main()
