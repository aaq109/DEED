  
"""
DEED EHR
@author: aaq109
"""

import argparse
from trainers import DEED_EHRtrainer

parser = argparse.ArgumentParser(description='DEED EHR arguments')

parser.add_argument('--fname', type=str, default='ehr_input',
                    help='Input pkl file name')
parser.add_argument('--iterations', type=int, default=10,
                    help='Training iterations')
parser.add_argument('--edl', default=True,
                    help='Use EDL loss; else use BCE loss')
parser.add_argument('--batch_size', type=int, default=128,
                    help='No. of patients per batch')
parser.add_argument('--sentence_limit', type=int, default=10,
                    help='Maximum number of visits/sentences allowed per patient/document')
parser.add_argument('--word_limit', type=int, default=20,
                    help='Maximum number of codes/words allowed per visit')
parser.add_argument('--emb_size', type=int, default=200,
                    help='Word embedding dimension')
parser.add_argument('--word_rnn_size', type=int, default=100,
                    help='Hidden layer size - word level')
parser.add_argument('--word_rnn_layers', type=int, default=2,
                    help='No. of hidden layers -  word level')
parser.add_argument('--sentence_rnn_size', type=int, default=100,
                    help='Hidden layer size - sentence level')
parser.add_argument('--sentence_rnn_layers', type=int, default=2,
                    help='No. of hidden layers -  sentence level')
parser.add_argument('--word_att_size', type=int, default=100,
                    help='Attention size - word level')
parser.add_argument('--sentence_att_size', type=int, default=100,
                    help='Attention size -  sentence level')
parser.add_argument('--dropout', type=int, default=0.5,
                    help='Drop out level')


args = parser.parse_args(args=[])
args.tied = True
dt = DEED_EHRtrainer(args)

dt.train_classifier(args)
dt.test_classifier()
if args.edl:
    dt.plot_classifier()

