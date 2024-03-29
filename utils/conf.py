import torch

def get_device(gpu_id):
    """
    Returns the GPU device if available else CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda:"+str(gpu_id))
    
    return torch.device("cpu")

def base_path():
    """
    Returns the base_path for the evaluation datasets
    """
    return './data/'
