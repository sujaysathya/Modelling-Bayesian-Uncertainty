import argparse, time, datetime
import pprint
import sys, os
import argparse

import numpy as np

import torch
import torchtext
from torch.utils.tensorboard import SummaryWriter
from nltk import word_tokenize
import nltk
# nltk.download('punkt')
from torch.autograd import Variable
from torchtext.data import Field, BucketIterator
from torchtext import datasets
import torch.nn as nn

from data_utils import *

def prepare_training():

    print("="*80 + "\n\t\t\t\t Preparing Data\n" + "="*80)

    print("\n\n==>> Getting Data splits..")
    train_docs, train_labels, test_docs, test_labels = get_data_splits()
    print("\nTotal training docs: {} \nTotal test docs: {}\n".format(len(train_docs), len(test_docs)) + "-"*35)

    print("\n==>> Tokenizing each document...")

    print("\nTraining set:\n")
    train_data = tokenize(train_docs)
    print("\nTest set:\n" + "-"*35)
    test_data = tokenize(test_docs)


    return None

def eval_network():

    return None


def train_network():

    return None






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required Paths
    parser.add_argument('--reuters_path', type = str, default = '../data/reuters',
                          help='path to reuters data (raw data)')
    parser.add_argument('--glove_path', type = str, default = '../data/glove/glove.840B.300d.txt',
                          help='path for Glove embeddings (850B, 300D)')
    parser.add_argument('--model_checkpoint_path', type = str, default = './model_checkpoints',
                          help='Directory for saving trained model checkpoints')
    parser.add_argument('--vis_path', type = str, default = './vis_checkpoints',
                          help='Directory for saving tensorboard checkpoints')
    parser.add_argument("--model_save_name", type=str, default= 'best_model.pt',
                       help = 'saved model name')

    # Training Params
    parser.add_argument('--model_name', type = str, default = 'bilstm',
                          help='model name: lstm / bilstm / bilstm_attn')
    parser.add_argument('--lr', type = float, default = 1e-4,
                          help='Learning rate for training')
    parser.add_argument('--batch_size', type = int, default = 32,
                          help='batch size for training"')
    parser.add_argument('--embed_dim', type = int, default = 300,
                          help='dimension of word embeddings used(GLove)"')
    parser.add_argument('--lstm_dim', type = int, default = 512,
                          help='dimen of hidden unit of LSTM/BiLSTM networks"')
    parser.add_argument('--fc_dim', type = int, default = 512,
                          help='dimen of FC layer"')
    parser.add_argument('--n_classes', type = int, default = 90,
                          help='number of classes"')
    parser.add_argument('--optimizer', type = str, default = 'Adam',
                        help = 'Optimizer to use for training')
    parser.add_argument('--embed_dpout', type = float, default = 0.3,
                        help = 'Embedding Dropout for training')
    parser.add_argument('--weight_dpout', type = float, default = 0.3,
                        help = 'Network weight Dropout for training')
    parser.add_argument('--weight_decay', type = float, default = 1e-4,
                        help = 'weight decay for optimizer')
    parser.add_argument('--momentum', type = float, default = 0.8,
                        help = 'Momentum for optimizer')
    parser.add_argument('--max_epoch', type = int, default = 50,
                        help = 'Max epochs to train for')

    args, unparsed = parser.parse_known_args()
    config = args.__dict__

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    # global dtype
    # dtype = torch.FloatTensor

    # Check all provided paths:
    model_path = os.path.join(config['model_checkpoint_path'], config['model_name'])
    if not os.path.exists(config['reuters_path']):
        raise ValueError("[!] ERROR: Reuters data path does not exist")
    if not os.path.exists(config['glove_path']):
        raise ValueError("[!] ERROR: Glove Embeddings path does not exist")
    if not os.path.exists(model_path):
        print("\nCreating checkpoint path for saved models at:  {}\n".format(model_path))
        os.makedirs(model_path)
    if not os.path.exists(args.vis_path):
        print("\nCreating checkpoint path for Tensorboard visualizations at:  {}\n".format(args.vis_path))
        os.makedirs(args.vis_path)
    if config['model_name'] not in ['lstm', 'bilstm', 'bilstm_attn']:
        raise ValueError("[!] ERROR:  model_name is incorrect. Choose one of - lstm / bilstm / bilstm_attn")


    # Prepare the tensorboard writer
    writer = SummaryWriter(os.path.join(args.vis_path, config['model_name']))

    # Prepare the datasets and iterator for training and evaluation
    # train_batch_loader, dev_batch_loader, test_batch_loader, TEXT, LABEL = prepare_training()
    prepare_training()

    #Print args
    print("\n" + "x"*50 + "\n\nRunning training with the following parameters: \n")
    for key, value in config.items():
        print(key + ' : ' + str(value))
    print("\n" + "x"*50)

    train_network()