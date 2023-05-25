"""
DEep Evidential Doctor - DEED
MNIST experiment
@author: awaash
"""


import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
from netcal.presentation import ReliabilityDiagram
from data_reader import DataReader_MNIST
from model import MNISTmodel, Decoder
from utils import get_device, euclid_dist, flatten, cosDist, epsilon, cal_ece
from losses import edl_loss


class DEEDtrainer:

    def __init__(self, args):

        self.zs_classes = len(args.zfs)
        self.zs = args.zfs
        self.lbl_enc_dim = args.lbl_enc_dim
        self.num_classes = args.num_classes
        self.use_attn = args.use_evidential_attention
        self.data = DataReader_MNIST(zfs=args.zfs,batch_size=args.batch_size)
        self.device = get_device()
        self.edl = args.edl
        self.neg_evd_w = args.neg_evd_w
        self.kpred = args.kpred
        self.model = MNISTmodel(self.num_classes,self.edl,self.use_attn).to(self.device) 
        self.iterations1 = args.iterations1
        self.iterations2 = args.iterations2
        self.class_uncertainty = {i:1 for i in range(self.num_classes)}
        self.uncertainty, self.uncertainty2 = [], []
        self.pred, self.pred2 = [], []
        self.emb = dict()
        self.classifier_out, self.regressor_out = [], []
        self.class_scores = dict()
        self.class_uncertainties = dict()
        self.batch_size = args.batch_size
        self.decoder = Decoder(args.num_classes, self.lbl_enc_dim,self.edl, [64],None)
        

    def train_classifier(self):
        optimizer = optim.Adam(self.model.parameters(),lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        criterion = [] if self.edl else nn.BCEWithLogitsLoss()
            
        for iteration in range(1,self.iterations1):
            losses = []
            self.model.train()
            for batch_idx, (data, target) in enumerate(tqdm(self.data.data_loaders['train'])):
                data = data.to(self.device)
                optimizer.zero_grad()
                if self.edl:
                    target_1hot = nn.functional.one_hot(target.long(), self.num_classes).sum(1).to(self.device).float()
                    output = self.model(data)
                    loss = edl_loss(output, target_1hot, self.neg_evd_w, device=self.device)               
                else:
                    target_1hot = nn.functional.one_hot(target.long(), self.num_classes).sum(1).to(self.device)
                    output = self.model(data)
                    loss = criterion(output,target_1hot.float()).float()
        
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print('Train losses:', np.mean(losses), 'Iteration',iteration, ' of ', self.iterations1)
            scheduler.step()
                    
                                    
    def test_classifier(self):
        #m = nn.Sigmoid()
        self.model.eval()
        outputs = np.empty([1,30])
        targets = np.empty([1,2])
        lbl_uncertainty = np.empty([1,30])

        for batch_idx, (data, target) in enumerate(tqdm(self.data.data_loaders['test'])):
            data = data.to(self.device)
            
            k = self.model(data)
            if self.edl:
                alpha = k + 1
                n = torch.arange(0,alpha.shape[1],2)
                m = torch.arange(1,alpha.shape[1],2)
                S = alpha[:,n] + alpha[:,m]
                scores_p = (alpha[:,n] / S).data.cpu().numpy()
                u = (alpha[:,n]* alpha[:,m]) / ((S +1)*(S*S))
                #u_norm = u / u.max(1, keepdim=True)[0]            

                outputs = np.append(outputs, scores_p, 0)
                targets = np.append(targets, target, 0)
                lbl_uncertainty = np.append(lbl_uncertainty, u.data.cpu().numpy(), 0)
            else:
                outputs = np.append(outputs, k.data.cpu().numpy(), 0)
                targets = np.append(targets, target.numpy(), 0)

        outputs = np.delete(outputs, (0), axis=0)
        targets = np.delete(targets, (0), axis=0)
            #lbl_uncertainty = np.delete(lbl_uncertainty, (0), axis=0)            
        self.classifier_out = (outputs,targets, lbl_uncertainty)
    
      
    def plot_classifier(self):
        y_prob, y_true, uncertainty = self.classifier_out[0], self.classifier_out[1], self.classifier_out[2]
        y_pred = y_prob.argsort(axis=1)[:,[-2,-1]]
        score = 0
        for i in range(y_true.shape[0]):
            score = score + len(np.intersect1d(y_true[i], y_pred[i]))

        print('Prediction_score:', score/(y_true.shape[0]*2))

        class_score = {i:0 for i in range(self.num_classes)}
        class_tot = {i:0 for i in range(self.num_classes)}

        for i,j in enumerate(y_true):
            for k in j:
                class_tot[k] = class_tot[k] + 1 
                if k in y_pred[i]:
                    class_score[k] = class_score[k] + 1
                    
        self.class_scores = {i:j/class_tot[i] for i,j in class_score.items()}
        print('Seen class accuracy:', np.mean([self.class_scores[i] for i in range(30) if i not in self.zs]))
        print('Unseen class accuracy:', np.mean([self.class_scores[i] for i in range(30) if i in self.zs]))
        print('Overall accuracy:', np.mean([self.class_scores[i] for i in range(30)]))
        if self.edl:
            uncertainty = np.delete(uncertainty, (0), axis=0)
            uncertainty = uncertainty / uncertainty.max(axis=0)
            class_uncertainty = {i:[] for i in range(self.num_classes)}
            for i,j in enumerate(y_true):
                ind = y_pred[i]
                u = uncertainty[i][ind]
                for k in j:
                    if k in ind:
                        class_uncertainty[k].append(u[np.where(ind==k)[0]][0])
                    else:
                        class_uncertainty[k].append(u[np.where(ind!=k)[0]][0])            

            self.class_uncertainties = {i:np.mean(j) for i,j in class_uncertainty.items()}

        if self.edl:
            plt.rcParams.update({'font.size': 22})
            plt.bar(self.class_scores.keys(), self.class_scores.values())
            plt.plot(self.class_uncertainties.keys(), self.class_uncertainties.values(),'--ro',label = 'Uncertainties')
            plt.legend()
            plt.title('Class scores with uncertainties')
            plt.xlabel('Classes')
            plt.ylabel('Accuracy / Uncertainties')
        else:
            plt.bar(self.class_scores.keys(), self.class_scores.values())
            plt.title('Class scores')
            plt.xlabel('Classes')
            plt.ylabel('Accuracy')
   
    def plot_ece(self, nbins=10):
        y_prob, y_true, uncertainty = self.classifier_out[0], self.classifier_out[1], self.classifier_out[2]
        uncertainty = np.delete(uncertainty, (0), axis=0)
        uncertainty = uncertainty / uncertainty.max(axis=0)

        discard_n = []
        for i,k in enumerate(y_true):
            if np.intersect1d(self.zs,k).size==0:
                discard_n.append(i)
                
        y_prob = y_prob[discard_n,:]
        y_true = y_true[discard_n,:]
        uncertainty = uncertainty[discard_n,:]
        
        y_pred = y_prob.argsort(axis=1)[:,[-2,-1]]
        y_correct_classified = []
        uncer_pred = []
        for ind,i in enumerate(y_pred):
            for k in i:
                if int(k) in y_true[ind]:
                    y_correct_classified.append(1)
                else: 
                    y_correct_classified.append(0) 

        y_correct_classified = np.array(y_correct_classified)
        uncer_pred = np.take_along_axis(uncertainty, y_pred, axis=1)  
        uncer_pred = 1-uncer_pred.ravel()
        self.ece_raw = pd.DataFrame(data=[uncer_pred,y_correct_classified]).T
        self.ece_raw.columns=['uncer','label']
        print('ECE = ',cal_ece(self.ece_raw, nbins))

        diagram = ReliabilityDiagram(nbins, equal_intervals=True)
        diagram.plot(uncer_pred,y_correct_classified, tikz=False, filename="edl_ece1.pdf")
        plt.close()
        diagram = ReliabilityDiagram(nbins, equal_intervals=False)
        diagram.plot(uncer_pred,y_correct_classified, tikz=False, filename="edl_ece2.pdf")
        plt.close()
   
    
    def train_regressor(self):             
        self.model.cpu()
        self.new_model = nn.Sequential(self.model,
                                       self.decoder
                                       )
        
        self.new_model.to(self.device)
        self.new_model.train()
        
        criterion = nn.MSELoss(size_average=False)
        optimizer = optim.Adam(self.new_model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        for iteration in range(1,self.iterations2):
            losses = []
            self.new_model.train()
            for batch_idx, (data, target) in enumerate(tqdm(self.data.data_loaders['train'])):
                data,target = data.to(self.device),target.numpy()
                optimizer.zero_grad()
                output = self.new_model(data)
                target_embeddings = torch.tensor([[self.data.emb_3d[j] for j in i] for i in target]).float().to(self.device)

                loss = criterion(output,target_embeddings)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print('Train losses:', np.mean(losses), 'Iteration',iteration)
            scheduler.step()
        
 
    def test_regressor(self):
        outputs = np.empty([1,2*self.kpred]) 
        targets = np.empty([1,2])
        out_raw = np.empty([1,3])
        tr = np.array(list(self.data.emb_3d.values()))
        self.new_model.eval()

        for batch_idx, (data, target) in enumerate(tqdm(self.data.data_loaders['test'])):
            x = target.shape[0]
            out = self.new_model(data.to(self.device))       
            out = out.detach().cpu().numpy()
            newarr = out.reshape(x*2,3)
            out_raw = np.append(out_raw,newarr,0)

            k = [flatten([list(np.argsort(euclid_dist(j,tr))[0:1*self.kpred]) for j in i]) for i in out]
            outputs = np.append(outputs, np.array(k), 0)
            targets = np.append(targets, target.numpy(), 0)
            
      
        outputs = np.delete(outputs, (0), axis=0)
        targets = np.delete(targets, (0), axis=0)
        out_raw = np.delete(out_raw, (0), axis=0)

        self.regressor_out = (outputs,targets,out_raw)
        
        
    def plot_regressor(self):
        y_pred, y_true, y_raw = self.regressor_out[0], self.regressor_out[1], self.regressor_out[2]
        score = 0
        for i in range(y_true.shape[0]):
            score = score + len(np.intersect1d(y_true[i], y_pred[i]))
        
        class_score = {i:[] for i in range(self.num_classes)}
        for i,j in enumerate(y_true):
            for k in j:
                if k in y_pred[i]:
                    class_score[k].append(1)
                else:
                    class_score[k].append(0)
            
        self.class_scores = {i:np.mean(j) for i,j in class_score.items()}
        print('Seen class accuracy:', np.mean([self.class_scores[i] for i in range(30) if i not in self.zs]))
        print('Unseen class accuracy:', np.mean([self.class_scores[i] for i in range(30) if i in self.zs]))
        print('Overall accuracy:', np.mean([self.class_scores[i] for i in range(30)]))
        
        bars = plt.bar(self.class_scores.keys(), self.class_scores.values())
        for i in self.zs:
            bars[i].set_color('r')
        plt.title('Class scores after regression')
        plt.xlabel('Classes')
        plt.ylabel('Accuracy')

        
        y_raw = pd.DataFrame([[i] for i in y_raw],columns = ["Estimated"])
        y_raw['Label'] = y_true.reshape(y_true.shape[0]*2,1)
        y_raw['Ground_truth'] = y_raw['Label'].map(self.data.emb_3d)
        y_raw = y_raw[y_raw['Label'].isin(self.zs)]
        y_raw['cosine_dist'] = y_raw.apply(lambda row: cosDist(row), axis=1)

        ecdf = ECDF(y_raw['cosine_dist'])
        x = np.round(np.linspace(0, 2, 10000),4)
        eps = epsilon(n=y_raw.shape[0])
        df_ecdf = pd.DataFrame(ecdf(x), index=x, columns=['ecdf'])      
        
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(figsize=(8,8))
        
        plt.plot(x, df_ecdf['ecdf'], 'b-')  
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.tick_params(bottom=False, left=False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#EEEEEE')
        ax.xaxis.grid(True, color='#EEEEEE')
        plt.xlabel('Distance from true label')
        plt.ylabel('Proportion of OOD examples')
        r = df_ecdf[df_ecdf.index==0.2].ecdf
        plt.plot([0.2 for i in range(1000)], np.linspace(0,r,1000),'grey',linestyle='--')
        plt.plot(np.linspace(0,0.2,1000), [r for i in range(1000)],'grey',linestyle='--')
        df_ecdf['upper'] = pd.Series(ecdf(x), index=x).apply(lambda x: min(x + eps, 1.))
        df_ecdf['lower'] = pd.Series(ecdf(x), index=x).apply(lambda x: max(x - eps, 0.))
        plt.fill_between(x, df_ecdf['upper'], df_ecdf['lower'], 
                         alpha=0.4, label='Confidence Band')
        plt.xlim(-0.1, 2)
        plt.ylim(0,1)
    

            
            
