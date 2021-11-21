"""
DEep Evidential Doctor - DEED
MNIST experiment
@author: awaash
"""

import argparse
from trainers import DEEDtrainer
import torch 

seed= 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='DEED arguments')

parser.add_argument('--zfs', default=[7,15,28],
                    help='zero frequency sample classes')
parser.add_argument('--iterations1', type=int, default=15,
                    help='Iterations for evidential classification step')
parser.add_argument('--iterations2', type=int, default=10,
                    help='Iterations for regression step')
parser.add_argument('--edl', default=True,
                    help='use edl output layer with edl_loss. Else linear output layer with BCE loss')
parser.add_argument('--batch_size', type=int, default=512,
                    help='Images per batch')
parser.add_argument('--num_classes', type=int, default=30,
                    help='All classes')
parser.add_argument('--lbl_enc_dim', default=3,
                    help='Label embedding dimension')
parser.add_argument('--kpred', default=1,
                    help='TopK predictions')
parser.add_argument('--neg_evd_w', default=0.1,
                    help='Scaling evidential loss if edl=True')

args = parser.parse_args(args=[])
args.tied = True


dt = DEEDtrainer(args)

dt.train_classifier()
dt.test_classifier()
dt.plot_classifier()

if args.edl:
    dt.train_regressor()
    dt.test_regressor()
    dt.plot_regressor()




