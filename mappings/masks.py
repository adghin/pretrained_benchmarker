"""
Benchmarks of pre-trained pytorch models on common continual learning datasets
@author: adrian.ghinea@outlook.it
"""

def groundTruth(dataset):
    """
    Get the ground truth for each dataset.
    """
    with open('/'+dataset+'.pkl','rb') as fp:
        ground_truth = pickle.load(fp)
    
    return ground_truth

def maskDataloader(image,label,dataset):
    """
    Creates a mask for the dataloader by removing images that doesn't have a mapping at all.
    Only for CIFAR10/100.
    """
    ground_truth = groundTruth(dataset)                                                #ground_truth contains matchings between CIFAR and Imagenet datasets

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

def maskSoftmax(tensor,dataset):
    """
    Mask the predicted tensor by removing predictions that doesn't map the target dataset.
    The argmax prediction will be taken on the new tensor of preds.
    :tensor: batch of image tensor
    :return: masked tensor
    """
    ground_truth = groundTruth(dataset)

    if isinstance(ground_truth,dict):   #CIFAR-10 & CIFAR-100                                            
        accepted_labels = sum([ground_truth[i] for i in ground_truth],[])       #list containing all classes that have a matching with cifar's labels
    if isinstance(ground_truth,list):   #Subsets of ImageNet
        accepted_labels = ground_truth
    
    for tens in tensor:
        for i,j in enumerate(tens):
            if(i not in accepted_labels):
                tens[i] = 0
    return tensor
