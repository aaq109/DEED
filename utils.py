"""
DEep Evidential Doctor - DEED
MNIST experiment
@author: awaash
"""

import torch
import numpy as np

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def one_hot_embedding(labels, num_classes=30):
    y = torch.eye(num_classes)
    return y[labels]

def euclid_dist(t1, t2):
    return np.sqrt(((t1-t2)**2).sum(axis = 1))


def flatten(t):
    return [item for sublist in t for item in sublist]