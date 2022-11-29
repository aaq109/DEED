"""
DEep Evidential Doctor - DEED
EHR experiment
@author: awaash
"""

import pickle

class DataReaderRHIP:
    
    def __init__(self, fname):

        with open(fname+'_train'+'.pkl', 'rb') as handle:
            self.data_train = pickle.load(handle)           

        with open(fname+'_test'+'.pkl', 'rb') as handle:
            self.data_test = pickle.load(handle) 
        

'''
The input data is a list of dictionaries. 
data = []
A dictionary is unique for each patient and build as follows:
pat = dict()
pat['Id'] = i   Unique patient identifier (int)
pat['text'] = [] List of list of all clinical codes. For instance, 
if a patient has two visits with codes a,b in first visit; and codes c,d in the second visit; then [[a,b],[c,d]]
pat['hdf'] = [] List of list of all clinical features. Prepared the same way as above.
pat['labels'] = [] List of all labels (str)
pat['catgy'] = [label2id[i] for i in pat['labels']] List of all label IDs 
data.append(pat)
save data as a pkl file with suffix _train and _test.
Of note, patients in train and test set should not be same
Also save label2id.pkl separately for later use
'''
            
   



