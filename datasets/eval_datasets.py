"""
### adrian.ghinea@outlook.it ###
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import datasets
from torch.utils.data import Dataset

class TinyImagenet(Dataset):
    """
    Class for the custom TinyImagenet testset.
    As this dataset doesn't give labels for the test set,
    the validation set will be used intead.

    All images will be transformed according to the default weights of
    the model used for the evaluation.
    """
    DS_NAME = 'tinyimagenet'
    
    def __init__(self,root,transform,download):
        """
        param: annotations_file (string): path to the csv file with annotations
        param: img_dir          (string): path to dir with images from validation set
        param: transform                : transform to be applied on a sample
        """        
        self.transform          = transform
        self.root               = root
        self.download           = download

        self.dataset_name       = 'tinyimagenet-nohd'
        self.path               = os.path.join(root,self.dataset_name)

        if download:
            if os.path.isdir(self.path) and len(os.listdir(self.path)) > 0:
                print("Dataset already downloaded")
            else:
                from onedrivedownloader import download
                print("Downloading dataset")
                ln = 'https://studentiunict-my.sharepoint.com/:u:/g/personal/ghndrn00t01z129z_studium_unict_it/EdZ5w35EkRJCuOHi5I9-pjIBI5BmjY9i3cGvEYkwiBcTtQ?e=J11g32'
                download(ln, filename=os.path.join(root, 'eval-tinyimagenet-nohd.zip'), unzip=True, unzip_path=root, clean=True)

        self.image_path         = os.path.join(self.path,'images')
        self.annotations_file   = os.path.join(self.path,'tinyimagenet_annotations.csv')
        self.img_labels         = pd.read_csv(self.annotations_file)

    def __len__(self):
        #To return the size of the dataset
        return len(self.img_labels)
    
    def __getitem__(self,idx):
        img_path        = os.path.join(self.image_path,self.img_labels.iloc[idx,0])

        #To return a PIL Image
        original_img    = Image.open(img_path).convert('RGB')                                 
        label           = self.img_labels.iloc[idx,1]                                        

        #Apply augmentation (if given)
        if self.transform is not None:
            image       = self.transform(original_img)
            return image,label
            
        return original_img,label

class TinyImagenetHD(Dataset):
    """
    Class for the custom TinyImagenet testset.
    As this dataset doesn't give labels for the test set,
    the validation set will be used intead.

    All images will be transformed according to the default weights of
    the model used for the evaluation.
    """
    DS_NAME = 'tinyimagenet-hd'
    
    def __init__(self,root,transform,download):
        """
        param: annotations_file (string): path to the csv file with annotations
        param: img_dir          (string): path to dir with images from validation set
        param: transform                : transform to be applied on a sample
        """        
        self.transform          = transform
        self.root               = root
        self.download           = download

        self.dataset_name       = 'tinyimagenet-hd'
        self.path               = os.path.join(root,self.dataset_name)

        if download:
            if os.path.isdir(self.path) and len(os.listdir(self.path)) > 0:
                print("Dataset already downloaded")
            else:
                from onedrivedownloader import download
                print("Downloading dataset")
                ln = 'https://studentiunict-my.sharepoint.com/:u:/g/personal/ghndrn00t01z129z_studium_unict_it/EVEsZyVoaCxCkNKCSzQAKkkBgayxsFFhFTu_AeZKyA1vug?e=M0UCla'
                download(ln, filename=os.path.join(root, 'eval-tinyimagenet-hd.zip'), unzip=True, unzip_path=root, clean=True)

        self.image_path         = os.path.join(self.path,'images')
        self.annotations_file   = os.path.join(self.path,'tinyimagenet_annotations.csv')
        self.img_labels         = pd.read_csv(self.annotations_file)

    def __len__(self):
        #To return the size of the dataset
        return len(self.img_labels)
    
    def __getitem__(self,idx):
        img_path        = os.path.join(self.image_path,self.img_labels.iloc[idx,0])

        #To return a PIL Image
        original_img    = Image.open(img_path).convert('RGB')                                 
        label           = self.img_labels.iloc[idx,1]                                        

        #Apply augmentation (if given)
        if self.transform is not None:
            image       = self.transform(original_img)
            return image,label
        
        return original_img,label

class ImagenetR(Dataset):
    """
    Class for the custom TinyImagenet testset.
    As this dataset doesn't give labels for the test set,
    the validation set will be used intead.

    All images will be transformed according to the default weights of
    the model used for the evaluation.
    """
    DS_NAME = 'imagenet-r'
    
    def __init__(self,root,transform,download):
        """
        param: annotations_file (string): path to the csv file with annotations
        param: img_dir          (string): path to dir with images from validation set
        param: transform                : transform to be applied on a sample
        """        
        self.transform          = transform
        self.root               = root
        self.download           = download
        
        self.dataset_name       = 'imagenet-r'
        self.path               = os.path.join(root,self.dataset_name)

        if self.download:
            if os.path.isdir(self.path) and len(os.listdir(self.path)) > 0:
                print("Dataset already downloaded")
            else:
                from onedrivedownloader import download
                print("Downloading dataset")
                ln = 'https://studentiunict-my.sharepoint.com/:u:/g/personal/ghndrn00t01z129z_studium_unict_it/ERF4aJ_1OM1NsrSwDO4i3xQB60a2N8M2crId6m7efgSl7A?e=v0KZTR'
                download(ln, filename=os.path.join(root, 'eval-imagenet-r.zip'), unzip=True, unzip_path=root, clean=True)

        self.image_path         = os.path.join(self.path,'images')
        self.annotations_file   = os.path.join(self.path,'imagenet-r_annotations.csv')
        self.img_labels         = pd.read_csv(self.annotations_file)

    def __len__(self):
        #To return the size of the dataset
        return len(self.img_labels)
    
    def __getitem__(self,idx):
        img_path        = os.path.join(self.image_path,self.img_labels.iloc[idx,0])

        #To return a PIL Image
        original_img    = Image.open(img_path).convert('RGB')                                 
        label           = self.img_labels.iloc[idx,1]                                        

        #Apply augmentation (if given)
        if self.transform is not None:
            image       = self.transform(original_img)
            return image,label
            
        return original_img,label

class CIFAR10(datasets.CIFAR10):
    """
    CIFAR10 dataset from torchvision
    """
    DS_NAME = 'cifar10'
    DS_MASK = DS_NAME
    
    def __init__(self,root,train,transform,download):
        self.root       = root
        self.transform  = transform
        super(CIFAR10, self).__init__(root, train, transform, download=not self._check_integrity())

    def __getitem__(self,idx):
        image, label    = self.data[idx], self.targets[idx]

        # to return a PIL Image
        image = Image.fromarray(image)
                                               
        #Apply augmentation (if given)
        if self.transform is not None:
            image = self.transform(image)
            
        return image,label

class CIFAR100(datasets.CIFAR100):
    """
    CIFAR100 dataset from torchvision
    """
    DS_NAME = 'cifar100'
    DS_MASK = DS_NAME
    
    def __init__(self,root,train,transform,download):
        self.root       = root
        self.transform  = transform
        super(CIFAR100, self).__init__(root, train, transform, download=not self._check_integrity())

    def __getitem__(self,idx):
        image, label    = self.data[idx], self.targets[idx]

        # to return a PIL Image
        image = Image.fromarray(image)                                          

        #Apply augmentation (if given)
        if self.transform is not None:
            image = self.transform(image)
            
        return image,label
