"""
DEep Evidential Doctor - DEED
MNIST experiment
@author: awaash
"""

#Prepare multi-label MNIST data set with backgrounds

from mnist_prep_utils import prepMNIST

print('Preparing train set')

prepMNIST('Data/mnist_train.csv',
          'Data/mnist_train_textured_multilabel.pkl',
          6000) #6000 samples per first label and then three backgrounds each (180,000 total images)

print('Preparing test set')
prepMNIST('Data/mnist_test.csv',
          'Data/mnist_test_textured_multilabel.pkl',
          2000) #2000 samples per first label and then three backgrounds each (6000 total images)
