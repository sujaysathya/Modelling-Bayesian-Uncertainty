import os, torch, random
import numpy as np
import nltk
from nltk.corpus import reuters
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")


def get_data_splits():
    train_docs, train_labels = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('training/')])
    test_docs, test_labels = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('test/')])
    return train_docs, train_labels, test_docs, test_labels

def split_train_set(config, train_data, train_labels):
    val_size = int(np.ceil(config['val_split'] * len(train_data)))
    train_size = int(len(train_data) - val_size)
    idx = list(range(len(train_data)))
    random.shuffle(idx)
    train_idx = idx[:train_size]
    val_idx = idx[train_size: train_size + val_size]
    train_x = [train_data[i] for i in train_idx]
    val_x = [train_data[i] for i in val_idx]
    train_y = [train_labels[i] for i in train_idx]
    val_y = [train_labels[i] for i in val_idx]
    return train_x, train_y, val_x, val_y

def tokenize(dataset):
    tokenized_docs = []
    for i in tqdm(range(len(dataset))):
        tokenized_docs.append(nltk.word_tokenize(dataset[i]))
    return tokenized_docs


def binarize_labels(class_list, train, val, test):
    labelencoder = MultiLabelBinarizer(classes = class_list)
    train = labelencoder.fit_transform(train)
    val = labelencoder.fit_transform(val)
    test = labelencoder.transform(test)
    print("\nTotal classes detected in each set: \n Train = {}, \n Val = {}, \n Test= {}".format(len(train[0]), len(val[0]), len(test[0])))
    return train, val, test


def build_vocab(data, glove_path):
    word_dict = {}
    embeddings = {}
    not_in_vocab = []
    for doc in data:
        for word in doc:
            if word not in word_dict:
                word_dict[word] = ''

    #start sentence token
    word_dict['<s>'] = ''
    
    #end sentence token
    word_dict['</s>'] = ''
    
    #unkown token for words not in GLOVE
    word_dict['unk'] = ''

    with open(glove_path, encoding="utf8") as f:
        for sents in f:
            word, emb = sents.split(' ',1)
            if word in word_dict:
                embeddings[word] = np.fromstring(emb, sep=' ')
#                 np.array(list(map(float, emb.split())))
    embeddings['unk'] = np.zeros(300)
    print("\nFound "+ str(len(embeddings)) + " words with Glove embeddings out of "+ str(len(word_dict)) + " total words in corpus.\n")
    return word_dict, embeddings


def get_batch_from_idx(config, word_emb, batch):
    sen_lens = np.array([len(x) for x in batch])
    max_len = np.max(sen_lens)
    embedded_sents = np.zeros((max_len, len(batch), config['embed_dim']))
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            if batch[i][j] in word_emb:
                embedded_sents[j, i, :] = word_emb[batch[i][j]]     # j,i and not i,j since we are working with batch_first = False for LSTMs
            else:
                embedded_sents[j, i, :] = word_emb['unk']
    return torch.from_numpy(embedded_sents).float(), torch.from_numpy(sen_lens)