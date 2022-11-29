"""
DEep Evidential Doctor - DEED
MNIST experiment
@author: awaash
"""

import torch
import torch.nn.functional as F

def edl_loss(output, target, neg_evd_w, device=None):
    fn = torch.digamma #From empirical experiments, torch.digamma is more stable than torch.log here
    alpha = output + 1
    evidence_split = torch.cat(torch.split(alpha,2,dim=1))
    target_split = torch.cat(torch.split(target,1,dim=1)) 
    target = 1 - F.one_hot(torch.flatten(target_split).long(), num_classes=2)
    S = torch.sum(evidence_split, dim=1, keepdim=True)
    A = torch.sum(target * (fn(S) - fn(evidence_split)), dim=1, keepdim=True)
    A = A*(target_split + neg_evd_w)
    return A.sum()/output.shape[0]
    
  
