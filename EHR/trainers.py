"""
DEep Evidential Doctor - DEED
EHR experiment
@author: awaash
"""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import json
import pickle
import seaborn as sns
from matplotlib import pyplot as plt

from data_extractor import DataReaderRHIP
from data_processor import DataProcessorRHIP
from model import HierarchialAttentionNetwork
from utils import compute_auc, HANDataset, load_word2vec_embeddings, edl_loss, adjust_learning_rate

seed= 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class DEED_EHRtrainer:

    def __init__(self, args):

        self.edl = args.edl
        self.data_raw = DataReaderRHIP(args.fname)
        self.hdf_len = len(self.data_raw.data_train[0]['hdf'][0])
        
        self.data_prosd = DataProcessorRHIP(args.sentence_limit, args.word_limit, args.emb_size, self.hdf_len)
        
        with open('word_map.json', 'r') as j:
            self.word_map = json.load(j)
        
        with open('label2id.pkl', 'rb') as handle:
            self.label2id  = pickle.load(handle)
        
        self.id2label = {j:i for i,j in self.label2id.items()}
        self.n_classes = len(self.id2label)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        
        self.train_loader = torch.utils.data.DataLoader(HANDataset('train'), batch_size=args.batch_size, shuffle=True,
                                              num_workers=8, pin_memory=True) 
        self.test_loader = torch.utils.data.DataLoader(HANDataset('test'), batch_size=1000, shuffle=False,
                                              num_workers=8, pin_memory=True)           
        self.model = HierarchialAttentionNetwork(n_classes=self.n_classes,
                                        vocab_size=len(self.word_map),
                                        emb_size=args.emb_size,
                                        word_rnn_size=args.word_rnn_size,
                                        sentence_rnn_size=args.sentence_rnn_size,
                                        word_rnn_layers=args.word_rnn_layers,
                                        sentence_rnn_layers=args.sentence_rnn_layers,
                                        word_att_size=args.word_att_size,
                                        sentence_att_size=args.sentence_att_size,
                                        edl=self.edl,
                                        hdf_len=self.hdf_len,
                                        dropout=args.dropout
                                        )
        embeddings, emb_size = load_word2vec_embeddings('word2vec_model', self.word_map)
        self.model.sentence_attention.word_attention.init_embeddings(embeddings) 
        self.model.sentence_attention.word_attention.fine_tune_embeddings(fine_tune=True)
        

    def train_classifier(self, args):
        optimizer = optim.Adam(self.model.parameters(),lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        self.model.to(self.device)
        self.model.train()
        for iteration in range(1,args.iterations):
            losses = []
            for i, (documents, sentences_per_document, words_per_sentence, labels, hdf, patients) in enumerate(self.train_loader):
                #i, (documents, sentences_per_document, words_per_sentence, labels, hdf, patients) = next(iter(enumerate(self.train_loader)))
                documents = documents.to(self.device)  # (batch_size, sentence_limit, word_limit)
                sentences_per_document = sentences_per_document.squeeze(1).to(self.device)  # (batch_size)
                words_per_sentence = words_per_sentence.to(self.device)  # (batch_size, sentence_limit)
                labels = labels.squeeze(1).to(self.device)  # (batch_size)
                hdf = hdf.squeeze().to(self.device)  # (batch_size, sentence_limit, 6)
                scores, word_alphas, sentence_alphas, _= self.model((documents, sentences_per_document,
                                                     words_per_sentence, hdf))  
                optimizer.zero_grad()
                if self.edl:
                    loss = edl_loss(scores, labels, device=self.device)
                else:
                    loss = criterion(scores, labels)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            print('Train losses:', np.mean(losses), 'Iteration',iteration, ' of ', args.iterations)
            adjust_learning_rate(optimizer, 0.99)

    def test_classifier(self):
        m = nn.Sigmoid()
        self.model.eval()
        all_scores, all_labels, all_uncertainties = [], [], []   
        self.model.to(self.device)

        for i, (documents, sentences_per_document, words_per_sentence, labels, hdf, patients) in enumerate(self.test_loader):
            #i, (documents, sentences_per_document, words_per_sentence, labels, hdf, patients) = next(iter(enumerate(self.test_loader)))
            documents = documents.to(self.device)  # (batch_size, sentence_limit, word_limit)
            sentences_per_document = sentences_per_document.squeeze(1).to(self.device)  # (batch_size)
            words_per_sentence = words_per_sentence.to(self.device)  # (batch_size, sentence_limit)
            labels = labels.squeeze(1).to(self.device)  # (batch_size)
            hdf = hdf.squeeze().to(self.device)  # (batch_size, sentence_limit, 6)                optimizer.zero_grad()
            scores, word_alphas, sentence_alphas, _= self.model((documents, sentences_per_document,
                                                     words_per_sentence, hdf))  
            if self.edl:
                alpha = torch.exp(scores) + 1
                n = torch.arange(0,alpha.shape[1],2)
                m = torch.arange(1,alpha.shape[1],2)
            
                scores = alpha[:,n] / (alpha[:,n] + alpha[:,m])
            
                S = alpha[:,n] + alpha[:,m]
                u = (alpha[:,n]* alpha[:,m]) / ((S +1)*(S*S))
                #u_norm = u / u.max(1, keepdim=True)[0] 
                
                all_uncertainties.extend(u.data.cpu().numpy().tolist())
            else:
                scores = m(scores)
            
            all_scores.extend(scores.data.cpu().numpy().tolist())
            all_labels.extend(labels.data.cpu().numpy().tolist())
                    
        roc_auc, fpr, tpr = compute_auc(np.array(all_scores), np.array(all_labels))
        class_uncertainty = {self.id2label[i]:[] for i,j in roc_auc.items()}

        if self.edl:     
            all_uncertainties = np.array(all_uncertainties)/ np.array(all_uncertainties).max(axis=0)
            all_uncertainties = all_uncertainties.tolist()            

            for i,j in enumerate(all_labels):
                targets = np.where(np.array(all_labels[i])==1)[0].tolist()
                n = len(targets)

                pred =  np.argsort(all_scores[i])[-n:]
                u = np.array(all_uncertainties[i])[pred]

                for k in targets:
                    if k in pred:
                        class_uncertainty[self.id2label[k]].append(u[np.where(pred==k)[0][0]])
                    else:
                        class_uncertainty[self.id2label[k]].append(np.max(u[np.where(pred!=k)[0]]))
        
        self.output = {i:(roc_auc[j],np.mean(class_uncertainty[i])) for i,j in self.label2id.items()}


    def plot_classifier(self):
        with open('prev_train_test.pkl', 'rb') as handle:
            temp = pickle.load(handle)
        prev_train, prev_test = temp[0], temp[1]
        df = pd.DataFrame(self.output).T.rename_axis('Categories').reset_index()
        df.columns = ['Categories','AUC','Uncertainty']
        df['prev_train'] = df['Categories'].map(prev_train)
        df['prev_test'] = df['Categories'].map(prev_test)

        plt.rcParams.update({'font.size': 18})
        sns.set_style("white")
        ax = sns.lmplot(x="prev_train", y="Uncertainty", hue=None, data=df, order=2,
                        height=5, aspect=1.6, robust=False, palette='tab10', 
                        scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))
        ax.set(xlim=(0.0, 25.0), ylim=(0, 1))
        plt.grid()  #just add this
        ax.set(xlabel='Train prevalence %', ylabel='Uncertainty')

        plt.show()
        plt.savefig('ehr_prev.pdf', format='pdf', dpi=500,bbox_inches='tight')

    
