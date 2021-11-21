"""
DEep Evidential Doctor - DEED
MNIST experiment
@author: awaash
Special thanks to shaohua0116 for the Multidigit MNIST code
"""

import pandas as pd
import numpy as np
import random
import pickle
from random import choice
from PIL import Image
from tqdm import tqdm 

def sample_coordinate(high, size):
    if high > 0:
        return np.random.randint(high, size=size)
    else:
        return np.zeros(size).astype(np.int)

def prepMNIST(input_file, output_file, N=6000):
    train_df = pd.read_csv(input_file)
    image = train_df.loc[:, train_df.columns != "label"].values.reshape((-1, 28, 28,1)) 
    label = train_df['label'].values


    h, w = image.shape[1:3]

    rs = np.random.RandomState(123)
    num_original_class = len(np.unique(label))
    num_class = len(np.unique(label))**2
    classes = list(np.array(range(num_class)))
    rs.shuffle(classes)
    
    indexes = []
    for c in range(num_original_class):
        indexes.append(list(np.where(label == c)[0]))

    image_size = [64,64]

    mnist2 = []
    mnist2_lbl = []
    
    print('Data loaded. Merging random digits for multilabel classification')
    
    for c in tqdm(range(10)):
        for k in range(N):
            digits = [c, choice([n for n in range(10) if n not in [c]])]

            imgs = [np.squeeze(image[np.random.choice(indexes[d])]) for d in digits]
            background = np.zeros((image_size)).astype(np.uint8)

            ys = sample_coordinate(image_size[0]-h, 2)
            xs = sample_coordinate(image_size[1]//2-w,size=2)
            xs = [l*image_size[1]//2+xs[l] for l in range(2)]

            for i in range(2):
                background[ys[i]:ys[i]+h, xs[i]:xs[i]+w] = imgs[i]
  
            mnist2.append(background.tolist())
            mnist2_lbl.append(digits)
   
    print('Digits merged. Adding random backgrounds')

    background_dict = {0:'vertical_64',1:'horizontal_64'}

    X_train_all = np.array(mnist2)
    y_train_all = np.array(mnist2_lbl)

    X_train_new = np.zeros((X_train_all.shape[0]*(len(background_dict)+1),64,64))
    y_train_new = np.zeros((y_train_all.shape[0]*(len(background_dict)+1),2))
    n = 0
    
    
    for i in tqdm(range(0,X_train_all.shape[0])):
        inp=X_train_all[i,:,:]
        img = Image.fromarray(inp.astype(np.uint8))
        X_train_new[n,:,:] = inp
        y_train_new[n,:] = y_train_all[i]
        n = n+1

        for j,k in background_dict.items():
            background = Image.open('Data/'+k+".png")
            background = Image.fromarray(np.asarray(background)* random.uniform(0.6,0.8))
            background = background.convert("L")
            new_img = Image.blend(background, img,  random.uniform(0.75,0.95))
            X_train_new[n,:,:] = np.asarray(new_img)
            y_train_new[n,:] = [int(str(j+1) + str(k)) for k in y_train_all[i]]
            #y_train_new[n,1] = j+1
            n +=1
    
    print('All done. Saving ..')
    
    with open(output_file, 'wb') as handle:
        pickle.dump((X_train_new,y_train_new), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('MNIST data file has been saved in {}.\n Data size: {}'.format(output_file,X_train_new.shape))



