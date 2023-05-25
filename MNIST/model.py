"""
DEep Evidential Doctor - DEED
MNIST experiment
@author: awaash
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class AttentionLayer(nn.Module):
    def __init__(self, d_in, n_classes, n_head, d_head=4, dropout=0.1, fuse=False):
        super().__init__()
        self.fuse = fuse
        self.n_classes = n_classes
        self.d_model = n_head * d_head
        self.class_emb = nn.Parameter(torch.zeros(n_classes, self.d_model))
        self.logits_w = nn.Parameter(torch.zeros(1, 1, 2))
        self.attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=n_head, batch_first=True)
        self.head = nn.Linear(self.d_model, 1)
        self.head0 = nn.Linear(d_in, n_classes)
        self.input_layer = nn.Linear(d_in, n_classes * self.d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.w_qs = nn.Linear(self.d_model, self.d_model)
        self.w_ks = nn.Linear(self.d_model, self.d_model)
        self.w_vs = nn.Linear(self.d_model, self.d_model)
        nn.init.xavier_normal_(self.class_emb)
        nn.init.xavier_normal_(self.logits_w) 


    def forward(self, x):
        y = self.input_layer(x)
        y = y.reshape(x.shape[0], self.n_classes, self.d_model)
        y = y + self.class_emb.unsqueeze(0)
        y = self.dropout(self.layer_norm(y))
        y, _ = self.attn(self.w_qs(y), self.w_ks(y), self.w_vs(y), need_weights=False)
        logits = self.head(y)
        if self.fuse:
            logits0 = self.head0(x)
            weighted_logits = torch.cat([logits0.unsqueeze(-1), logits], axis=-1) * torch.softmax(self.logits_w, axis=-1)
            logits = torch.logsumexp(weighted_logits, axis=-1)
        else:
            logits = logits[..., 0]
        return logits

class Evidential_layer(nn.Module):
    def __init__(self, in_dim, num_classes, use_attn=True):
        super(Evidential_layer, self).__init__()
        self.num_classes = num_classes
        self.use_attn = use_attn
        if use_attn:
            self.fc1 = AttentionLayer(in_dim, 2 * self.num_classes, n_head=4, d_head=4, dropout=0.1)
        else:
            self.fc1 = nn.Linear(in_dim, 2 * self.num_classes)
        self.relu = torch.nn.ReLU()

    @staticmethod
    def evidence_fn(logits):
        x = torch.exp(torch.clamp(logits, max=10))
        y = torch.nn.functional.relu(logits)
        evidence = x + (y - y.detach())
        #print((y - y.detach()).sum())
        return evidence

    def forward(self, x):
        x = self.fc1(x)
        if self.use_attn:
            return self.evidence_fn(x)
        else:
            return self.relu(x)


class MNISTmodel(nn.Module):
    def __init__(self, num_classes, edl, use_attn, dropout=True):
        super(MNISTmodel, self).__init__()

        self.use_dropout = dropout
        self.use_attn = use_attn
        k,m=8,80
        km = (64-(2*(k-1)))**2*m
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 20, kernel_size=k)
        self.conv2 = nn.Conv2d(20, m, kernel_size=k)
        self.fc1 = nn.Linear(km, 500)
        if edl:
            self.fc2 = Evidential_layer(500, self.num_classes, use_attn=self.use_attn)
        else:
            self.fc2 = nn.Linear(500, self.num_classes)      

    def forward(self, x, p=0.5):
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        x = F.relu(F.max_pool2d(self.conv2(x), 1))

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training, p=p)
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
