"""
DEep Evidential Doctor - DEED
MNIST experiment
@author: awaash
"""

import numpy as np
import torch
import pickle
import sklearn


class DataReader_MNIST:

    def __init__(self, zfs=[], batch_size=1000):
        self.zfs = zfs
        self.batch_size = batch_size
        self.data_loaders = dict()
        self.mnist = dict()
        self.emb_3d = dict()
        self.read_mnist()

    def read_mnist(self):
        with open('Data/mnist_train_textured_multilabel.pkl', 'rb') as handle:  
            temp = pickle.load(handle)
        X_train, y_train = temp[0], np.sort(temp[1])
                                            
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=0)
        X_train = X_train.astype('float64')/255

        
        with open('Data/mnist_test_textured_multilabel.pkl', 'rb') as handle:  
            temp = pickle.load(handle)
        X_test, y_test = temp[0], np.sort(temp[1])
        X_test = X_test.astype('float64')/255

        

        if len(self.zfs)>0:
            temp = []
            for i,j in enumerate(y_train):
                if len(np.intersect1d(self.zfs, j))>0:
                    temp.append(i)
            X_train = np.delete(X_train, temp, axis=0)
            y_train = np.delete(y_train, temp, axis=0)
            print(len(temp))
  
        
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.Tensor(np.expand_dims(X_train, axis=1)).float(),
                torch.Tensor(y_train).float(),
                ),
            batch_size=self.batch_size,
            shuffle=False,
            )


        test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.Tensor(np.expand_dims(X_test, axis=1)).float(),
                    torch.Tensor(y_test).float(),
                    ),
                batch_size=self.batch_size,
                shuffle=False,
                )
        
        self.data_loaders['train'] = train_loader
        self.data_loaders['test'] = test_loader
        
        with open('Data/emb.pkl', 'rb') as handle:
            self.emb_3d = pickle.load(handle)
        
        print('Training size = ', X_train.shape)
        print('Test size = ', X_test.shape)


 
