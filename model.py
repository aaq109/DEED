"""
DEep Evidential Doctor - DEED
MNIST experiment
@author: awaash
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class Evidential_layer(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(Evidential_layer, self).__init__()
        
        self.num_classes = num_classes
        self.fc1 = nn.Linear(in_dim, 2*self.num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)

        return self.relu(x)  #Can also use torch.exp for non-negative constraint

class MNISTmodel(nn.Module):
    def __init__(self, num_classes,edl, dropout=True):
        super(MNISTmodel, self).__init__()

        self.use_dropout = dropout
        k,m=8,80
        km = (64-(2*(k-1)))**2*m
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 20, kernel_size=k)
        self.conv2 = nn.Conv2d(20, m, kernel_size=k)
        self.fc1 = nn.Linear(km, 500)
        if edl:
            self.fc2 = Evidential_layer(500, self.num_classes)
        else:
            self.fc2 = nn.Linear(500, self.num_classes)      

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        x = F.relu(F.max_pool2d(self.conv2(x), 1))

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim,edl, h_dims, h_activ):
        super(Decoder, self).__init__()

        self.edl = edl
        layer_dims = [input_dim] + h_dims + [h_dims[-1]]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True
            )
            self.layers.append(layer)
        self.h_activ = h_activ
        self.dense_matrix = nn.Parameter(
            torch.rand((layer_dims[-1], out_dim), dtype=torch.float),
            requires_grad=True)

    def forward(self, x, seq_len=2):
        if self.edl:
            alpha = x + 1
            n = torch.arange(0,x.shape[1],2)
            m = torch.arange(1,alpha.shape[1],2)
            S = alpha[:,n] + alpha[:,m]
            x = (alpha[:,n] / S)
        
        x = torch.repeat_interleave(x.unsqueeze(1), repeats=seq_len, dim=1)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)
            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
        
        return torch.matmul(x, self.dense_matrix.data)