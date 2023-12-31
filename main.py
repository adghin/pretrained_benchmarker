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

import timm
import torch
import torchvision as tv
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader

def parseArgs():
    """
    Arguments to use: model_name, dataset_name, batch_size
    """
    parser = argparse.ArgumentParser(description="Benchmark of pre-trained models on common continual learning datasets")

    parser.add_argument("-m"      , "--model", type=str, help="Available models: resnet18, resnet34, resnet50, resnet101, resnet152, vit_b_16, vit_b_32, vit_l_16",required=True)
    parser.add_argument("-d"      , "--dataset", type=str, help="Available datasets: cifar10, cifar100, tinyimagenet",required=True)
    parser.add_argument("-b"      , "--batch_size", type=int, help="batch size for testing (default=32)",default=32)
    parser.add_argument("-gpu"    , "--gpu", type=int, help="Choose on which GPU to run the experiment",default=0)
    parser.add_argument("-md"     , "--mask_dataloader", type=bool, help="Remove datasets' samples that doesn't have any matching with the dataset on which the model was pretrained on (FOR CIFAR-10 & CIFAR-100)", default=False)
    parser.add_argument("-preproc", "--preprocess", type=str, help="Use RESNET_CUSTOM for a custom preprocess transform on the dataset (ResNet & CIFAR)")

    args = parser.parse_args()

    return args

def get_device():
    """
    Returns the GPU device if available else CPU.
    """
    gpu_id = args.gpu
    
    if torch.cuda.is_available():
        return torch.device("cuda:"+str(gpu_id))
    
    return torch.device("cpu")

def model():
    """
    Load pre-trained model with default weights from torchvision.models
    """
    args = parseArgs()
    model_name = args.model
    model_weights = "DEFAULT"
    model = models.get_model(model_name,weights=model_weights)

    return model

def dataset():
    """
    Load test dataset
    """
    args = parseArgs()
    dataset_name = args.dataset.upper()

    transform_weights = models.get_model_weights(args.model)
    default_weights = transform_weights.DEFAULT

    if(parseArgs().preprocess == "RESNET_CUSTOM"):
        preprocess = transforms.Compose([
                            transforms.Resize(132, interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.CenterCrop(128),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                            ])
    else:
        preprocess= default_weights.transforms()

    if hasattr(datasets,dataset_name):
        #For CIFAR-10 and CIFAR-100
        imported_dataset = getattr(datasets,dataset_name)
        test_dataset     = imported_dataset(root='./data',train=False,transform=preprocess,download=True)
    else:
        #For TinyImagenet
        imported_module  = importlib.import_module('.'+parseArgs().dataset,'utils')
        annotation_path  = 'utils/tinyimagenet_annotations.csv'
        image_dir_path   = 'utils/tinyimagenet-nohd/images'
        test_dataset     = imported_module.MyTinyImagenet(root=image_dir_path,annotations_file=annotation_path,transform=default_weights.transforms())

    dataloader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,drop_last=False)
    print(len(dataloader))

    return dataloader

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()   
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    plt.show()

    #Use this function by calling imshow(tv.utils.make_grid(image))

def groundTruth(dataset):
    """
    Get the ground truth for each datasetclear
    """
    with open('utils/'+dataset+'.pkl','rb') as fp:
        ground_truth = pickle.load(fp)
    
    return ground_truth

def maskDataloader(image,label):
    """
    This function creates a mask for the dataloader thus it removes images that doesn't have a mapping at all only for CIFAR-10 & CIFAR-100
    """
    ground_truth = groundTruth(parseArgs().dataset)                             #ground_truth contains matchings between CIFAR and Imagenet datasets

    image_list = [idx for idx in image]                                         #from tensor to list of tensors
    label_list = label.tolist()                                                 #from tensor to list

    remove_idxs = []                                                            #idxs of images to be removed

    empty_list  = [i for i in ground_truth if len(ground_truth[i]) == 0]        #labels with no matching

    for i,j in enumerate(label_list):
        if(j in empty_list):
            remove_idxs.append(i)    

    image_list = [i for j,i in enumerate(image_list) if j not in remove_idxs]   #get only valid images (i.e. remove wrong idxs)
    label_list = [i for j,i in enumerate(label_list) if j not in remove_idxs]   #get only valid labels (i.e. remove wrong idxs)
        
    image_tensor = torch.stack(image_list,0)                                    #re-convert the list of image tensors to tensor
    label_tensor = torch.tensor(label_list)                                     #re-convert the list of labels to tensor

    return image_tensor,label_tensor

def maskSoftmax(tensor):
    """
    Mask the softmax predicted tensor by removing predictions that could be wrong
    """
    ground_truth = groundTruth(parseArgs().dataset)

    if isinstance(ground_truth,dict):   #CIFAR-10 & CIFAR-100                                            
        accepted_labels = sum([ground_truth[i] for i in ground_truth],[])       #list containing all classes that have a matching with cifar's labels
    if isinstance(ground_truth,list):   #TinyImagenet
        accepted_labels = ground_truth
    
    for tens in tensor:
        for i,j in enumerate(tens):
            if(i not in accepted_labels):
                tens[i] = 0
    return tensor

def evaluateModel(model):
    """
    Evaluate the model with accuracy on testset (correct_images/total_images)
    """
    args = parseArgs()
    if(args.dataset == 'cifar10' or args.dataset == 'cifar100'):
        ground_truth = groundTruth(args.dataset)                                            #ground_truth contains matchings between CIFAR and Imagenet datasets
    
    test_loader = dataset()
    model.eval()

    correct = 0
    total   = 0
    
    with torch.no_grad():
        for image,label in tqdm(test_loader):

            if((args.mask_dataloader == True)):
                assert args.dataset == 'cifar10' or args.dataset == 'cifar100', "A dataloader mask can be applied only on CIFAR-10 and CIFAR-100 dataset!"

                masked_image, masked_label = maskDataloader(image,label)
                image = masked_image
                label = masked_label

            prediction = model(image).softmax(dim=1)                                        #tensor with predictions for each sample
            
            masked_prediction = maskSoftmax(prediction)                                     #remove unaccepted labels (i.e. labels of iamgenet that doesn't have a match with any label from cifars)

            pred_class_id = [idx.argmax().item() for idx in masked_prediction]              #get class_id from predictions

            
            ground_labels = [idx.item() for idx in label]                                   #ground_labels: list of truth labels

            for i, label in enumerate(pred_class_id):
                total += 1
                if(args.dataset == 'cifar10' or args.dataset == 'cifar100'):                #CIFARS --> use ground_truth
                    if(label in ground_truth[ground_labels[i]]):
                        correct += 1
                else:
                    if(label == ground_labels[i]):                                          #Tiny-ImageNet --> use ground_labels directly
                        correct += 1

    
    print("\n\nCorrect: " + str(correct))
    print("\nTotal: " + str(total))
    print('\nTest Accuracy: {:.2f}%'.format(100*correct/total))

def main(args=None):
    #my_model = model()
    avail_pretrained_models = timm.list_models(pretrained=True)
    print(avail_pretrained_models)
    pretrained_vitb16_in21k = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
    evaluateModel(pretrained_vitb16_in21k)

main()
