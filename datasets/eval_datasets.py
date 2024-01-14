"""
### adrian.ghinea@outlook.it ###
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple

from torchvision import datasets
from torchvision.utils.data import Dataset

import torchvision.transforms as transforms

class TinyImagenet(Dataset):
    """
    Class for the custom TinyImagenet testset.
    As this dataset doesn't give labels for the test set,
    the validation set will be used intead.

    All images will be transformed according to the default weights of
    the model used for the evaluation.
    """
    def __init__(self,root='./data',transform,download):
        """
        param: annotations_file (string): path to the csv file with annotations
        param: img_dir          (string): path to dir with images from validation set
        param: transform                : transform to be applied on a sample
        """        
        self.transform          = transform
        self.root               = root
        sef.download            = download

        if self.download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print("Dataset already downloaded")
            else:
                print("Downloading dataset")
                ln = 'https://studentiunict-my.sharepoint.com/:u:/g/personal/ghndrn00t01z129z_studium_unict_it/EdZ5w35EkRJCuOHi5I9-pjIBI5BmjY9i3cGvEYkwiBcTtQ?e=J11g32'
                download(ln, filename=os.path.join(root, 'eval-tiny-imagenet-nohd.zip'), unzip=True, unzip_path=root, clean=True)

        self.image_path         = os.path.join(root,'tinyimagenet-nohd/images')
        self.annotations_file   = os.path.join(root,'tinyimagenet-nohd/tinyimagenet_annotations.csv')
        self.img_labels         = pd.read_csv(self.annotations_file)

    def __len__(self):
        #To return the size of the dataset
        return len(self.img_labels)
    
    def __getitem__(self,idx) -> Tuple[image,label]:
        img_path        = os.path.join(self.image_path,self.img_labels.iloc[idx,0])

        #To return a PIL Image
        original_img    = Image.open(img_path).convert('RGB')                                 
        label           = self.img_labels.iloc[idx,1]                                        

        #Apply augmentation (if given)
        if self.transform is not None:
            image       = self.transform(image)
            
        return image,label

class TinyImagenetHD(Dataset):
    """
    Class for the custom TinyImagenet testset.
    As this dataset doesn't give labels for the test set,
    the validation set will be used intead.

    All images will be transformed according to the default weights of
    the model used for the evaluation.
    """
    def __init__(self,root='./data',transform,download):
        """
        param: annotations_file (string): path to the csv file with annotations
        param: img_dir          (string): path to dir with images from validation set
        param: transform                : transform to be applied on a sample
        """        
        self.transform          = transform
        self.root               = root
        sef.download            = download

        if self.download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print("Dataset already downloaded")
            else:
                print("Downloading dataset")
                ln = 'https://studentiunict-my.sharepoint.com/:u:/g/personal/ghndrn00t01z129z_studium_unict_it/EVEsZyVoaCxCkNKCSzQAKkkBgayxsFFhFTu_AeZKyA1vug?e=M0UCla'
                download(ln, filename=os.path.join(root, 'eval-tiny-imagenet-hd.zip'), unzip=True, unzip_path=root, clean=True)

        self.image_path         = os.path.join(root,'tinyimagenet-hd/images')
        self.annotations_file   = os.path.join(root,'tinyimagenet-hd/tinyimagenet_annotations.csv')
        self.img_labels         = pd.read_csv(self.annotations_file)

    def __len__(self):
        #To return the size of the dataset
        return len(self.img_labels)
    
    def __getitem__(self,idx) -> Tuple[image,label]:
        img_path        = os.path.join(self.image_path,self.img_labels.iloc[idx,0])

        #To return a PIL Image
        original_img    = Image.open(img_path).convert('RGB')                                 
        label           = self.img_labels.iloc[idx,1]                                        

        #Apply augmentation (if given)
        if self.transform is not None:
            image       = self.transform(image)
            
        return image,label

class TinyImagenetR(Dataset):
    """
    Class for the custom TinyImagenet testset.
    As this dataset doesn't give labels for the test set,
    the validation set will be used intead.

    All images will be transformed according to the default weights of
    the model used for the evaluation.
    """
    def __init__(self,root='./data',transform,download):
        """
        param: annotations_file (string): path to the csv file with annotations
        param: img_dir          (string): path to dir with images from validation set
        param: transform                : transform to be applied on a sample
        """        
        self.transform          = transform
        self.root               = root
        sef.download            = download

        if self.download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print("Dataset already downloaded")
            else:
                print("Downloading dataset")
                ln = 'https://studentiunict-my.sharepoint.com/:u:/g/personal/ghndrn00t01z129z_studium_unict_it/EZ9f00uX7EtJjaJxgbsmfk4BJ-VRgmWbpYeVnmYGnuLd1Q?e=Nyw26q'
                download(ln, filename=os.path.join(root, 'eval-tiny-imagenet-r.zip'), unzip=True, unzip_path=root, clean=True)

        self.image_path         = os.path.join(root,'tinyimagenet-r/images')
        self.annotations_file   = os.path.join(root,'tinyimagenet-r/tinyimagenet-r_annotations.csv')
        self.img_labels         = pd.read_csv(self.annotations_file)

    def __len__(self):
        #To return the size of the dataset
        return len(self.img_labels)
    
    def __getitem__(self,idx) -> Tuple[image,label]:
        img_path        = os.path.join(self.image_path,self.img_labels.iloc[idx,0])

        #To return a PIL Image
        original_img    = Image.open(img_path).convert('RGB')                                 
        label           = self.img_labels.iloc[idx,1]                                        

        #Apply augmentation (if given)
        if self.transform is not None:
            image       = self.transform(image)
            
        return image,label

class CIFAR10(datasets.CIFAR10):
    def __init__(self, root='./data',train=False,transform,download):
        
        self.transform  = transform
        self.download   = download
        super(CIFAR10, self).__init__(root, train, transform, download=not self._check_integrity())

    def __len__(self):
        #To return the size of the dataset
        return len(self.img_labels)

    def __getitem__(self,idx) -> Tuple[image,label]:
        imgage, label      = self.data[idx], self.targets[idx]

        #To return a PIL Image
        image   = Image.fromarray(image)                                                 

        #Apply augmentation (if given)
        if self.transform is not None:
            image = self.transform(image)
            
        return image,label

class CIFAR100(Dataset):
    def __init__(self, root='./data',train=False,transform,download):
        
        self.transform  = transform
        self.download   = download
        super(CIFAR10, self).__init__(root, train, transform, download=not self._check_integrity())

    def __len__(self):
        #To return the size of the dataset
        return len(self.img_labels)

    def __getitem__(self,idx) -> Tuple[image,label]:
        imgage, label      = self.data[idx], self.targets[idx]

        #To return a PIL Image
        image   = Image.fromarray(image)                                                 

        #Apply augmentation (if given)
        if self.transform is not None:
            image = self.transform(image)
            
        return image,label
