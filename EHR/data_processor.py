"""
DEep Evidential Doctor - DEED
EHR experiment
@author: awaash
"""

import pickle
from tqdm import tqdm
import json
from collections import Counter
import torch
import gensim
import itertools
import logging


class DataProcessorRHIP:

    def __init__(self, sentence_limit, word_limit, emb_size, hdf_len, pretrain=True):
        
        self.word_map = dict()
        self.id2label = dict()
        self.hdf_len = hdf_len
        self.prep_Data('train', sentence_limit, word_limit,w2v=True)
        self.prep_Data('test', sentence_limit, word_limit,w2v=False)
        if pretrain:
            self.train_word2vec_model(emb_size, algorithm="skipgram")

    def prep_Data(self, tp, sentence_limit, word_limit, w2v, min_word_count=10):
    # Read training data
        print('\nReading and preprocessing {0} data...\n'.format(tp))
        with open('ehr_input'+'_'+tp+'.pkl', 'rb') as handle:
            train = pickle.load(handle)
        with open('label2id.pkl', 'rb') as handle:
            label2id  = pickle.load(handle)
        
    
        docs = []
        labels = []
        patients = []
        hdf = []
        word_counter = Counter()
    
        for i in tqdm(train):
            sentences = i['text']
            words = list()
            for s in sentences[-sentence_limit:]:
                w = s[-word_limit:]
                if len(w) == 0:
                    continue
                words.append(w)
                word_counter.update(w)
        # If all sentences were empty
            if len(words) == 0:
                continue

            docs.append(words)
            
            x = torch.nn.functional.one_hot(torch.tensor(i['catgy']), len(label2id)).sum(0).numpy().tolist()        
            labels.append(x)
            patients.append(i['Id'])
            hdf.append(i['hdf'])
            #labels.append(i['labels'])
    

    # Save text data for word2vec
        if w2v:
            torch.save(docs, 'word2vec_data.pth.tar')
            print('\nText data for word2vec saved')

    # Create word map
            self.word_map['<pad>'] = 0
            for word, count in word_counter.items():
                if count >= min_word_count:
                    self.word_map[word] = len(self.word_map)
            self.word_map['<unk>'] = len(self.word_map)
            print('\nDiscarding words with counts less than %d, the size of the vocabulary is %d.\n' % (
                min_word_count, len(self.word_map)))

            with open('word_map.json', 'w') as j:
                json.dump(self.word_map, j)
            print('Word map saved')

    # Encode and pad
        print('Encoding and padding training data...\n')
        encoded_train_docs = list(map(lambda doc: list(
            map(lambda s: list(map(lambda w: self.word_map.get(w, self.word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
                doc)) + [[0] * word_limit] * (sentence_limit - len(doc)), docs))
        sentences_per_train_document = list(map(lambda doc: len(doc), docs))
        words_per_train_sentence = list(
            map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (sentence_limit - len(doc)), docs))
        temp = [hdf[i][-j:] for i,j in enumerate(sentences_per_train_document)]
        encoded_train_hdf = list(
            map(lambda doc: list(map(lambda s: s, doc)) + [[0] * self.hdf_len]* (sentence_limit - len(doc)), temp))

    # Save
        print('Saving...\n')
        assert len(encoded_train_docs) == len(labels) == len(sentences_per_train_document) == len(
            words_per_train_sentence)
    # Because of the large data, saving as a JSON can be very slow
        torch.save({'docs': encoded_train_docs,
                    'labels': labels,
                    'sentences_per_document': sentences_per_train_document,
                    'words_per_sentence': words_per_train_sentence,
                    'hdf': encoded_train_hdf,
                    'patients': patients},
                   tp+'_data.pth.tar')
        print('Encoded, padded {0} data'.format(tp))


    def train_word2vec_model(self, emb_size, algorithm='skipgram'):
        assert algorithm in ['skipgram', 'cbow']
        sg = 1 if algorithm is 'skipgram' else 0

        # Read data
        sentences = torch.load('word2vec_data.pth.tar')
        sentences = list(itertools.chain.from_iterable(sentences))

        # Activate logging for verbose training
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # Initialize and train the model (this will take some time)
        model = gensim.models.word2vec.Word2Vec(sentences=sentences, vector_size=emb_size, workers=8, window=20, min_count=10,
                                            sg=sg)

        # Normalize vectors and save model
        model.init_sims(True)
        model.wv.save('word2vec_model')






