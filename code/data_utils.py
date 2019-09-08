import os, torch
import numpy as np
import nltk
from nltk.corpus import reuters
from tqdm import tqdm


def get_data_splits():

    train_docs, train_labels = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('training/')])
    test_docs, test_labels = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('test/')])

    return train_docs, train_labels, test_docs, test_labels

def tokenize(dataset):

    tokenized_docs = []
    for i in tqdm(range(len(dataset))):
        tokenized_docs.append(nltk.word_tokenize(dataset[i]))

    return tokenized_docs
