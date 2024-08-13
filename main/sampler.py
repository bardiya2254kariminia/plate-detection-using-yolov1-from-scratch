import torch
from torch.utils.data import WeightedRandomSampler
def get_sampler(dataset):
    
    sample_weights = [0] * len(dataset)
    for idx , (x,y) in enumerate(dataset):
        if y[...,0].sum() == 0:
            sample_weights[idx] = 1/4952
        else:
            sample_weights[idx] = 1/433
    sampler = WeightedRandomSampler(
        weights=sample_weights ,
        num_samples=len(dataset),
        replacement=True
    )
    return sampler
    