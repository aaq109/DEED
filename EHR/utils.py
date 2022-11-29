  
"""
DEED EHR
@author: aaq109
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
import gensim
from sklearn.metrics import roc_curve, auc


def compute_auc(x,y):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(x.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], x[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc, fpr, tpr

def relu_evidence(y):
    return F.relu(y)


def edl_loss(output, target,device=None):
    fn = torch.digamma
    evidence = torch.exp(output) + 1
    
    evidence_split = torch.cat(torch.split(evidence,2,dim=1))
    target_split = torch.cat(torch.split(target,1,dim=1)) 
    target = 1 - F.one_hot(torch.flatten(target_split).long(), num_classes=2)
    S = torch.sum(evidence_split, dim=1, keepdim=True)
    A = torch.sum(target* (fn(S) - fn(evidence_split)), dim=1, keepdim=True)
    x = target_split + 0.05
    A = A*x
    return A.sum()/output.shape[0]


def init_embedding(input_embedding):
    """
    Initialize embedding tensor with values from the uniform distribution.

    :param input_embedding: embedding tensor
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def load_word2vec_embeddings(word2vec_file, word_map):
    """
    Load pre-trained embeddings for words in the word map.

    :param word2vec_file: location of the trained word2vec model
    :param word_map: word map
    :return: embeddings for words in the word map, embedding size
    """
    # Load word2vec model into memory
    w2v = gensim.models.KeyedVectors.load(word2vec_file, mmap='r')

    print("\nEmbedding length is %d.\n" % w2v.vector_size)

    # Create tensor to hold embeddings for words that are in-corpus
    embeddings = torch.FloatTensor(len(word_map), w2v.vector_size)
    init_embedding(embeddings)

    # Read embedding file
    print("Loading embeddings...")
    for word in word_map:
        if word in w2v.key_to_index:
            embeddings[word_map[word]] = torch.FloatTensor(w2v[word])

    print("Done.\n Embedding vocabulary: %d.\n" % len(word_map))

    return embeddings, w2v.vector_size

def adjust_learning_rate(optimizer, scale_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rates must be decayed
    :param scale_factor: factor to scale by
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

class HANDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        """
        assert split in {'train', 'test'}
        self.split = split

        # Load data
        self.data = torch.load(split+'_data.pth.tar')


    def __getitem__(self, i):
        
        return torch.LongTensor(self.data['docs'][i]), \
                      torch.LongTensor([self.data['sentences_per_document'][i]]), \
                      torch.LongTensor(self.data['words_per_sentence'][i]), \
                      torch.FloatTensor([self.data['labels'][i]]), \
                      torch.FloatTensor([self.data['hdf'][i]]), \
                      torch.LongTensor([self.data['patients'][i]])


    def __len__(self):
        return len(self.data['docs'])



