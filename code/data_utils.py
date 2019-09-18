import os, torch, random, re, torchtext
import numpy as np
import nltk
from nltk.corpus import reuters
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from torchtext.data import TabularDataset, Field, NestedField, BucketIterator
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#####################################
## Older data pre-prcoessing files ##
#####################################

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

def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    tokens = string.lower().strip().split()
    return tokens

def clean_string_stop_words_remove(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    tokens = string.lower().strip().split()
    tokens = [t for t in tokens if not t in nltk_stopwords]
    return tokens


def split_sents(string):
    string = re.sub(r"[!?]"," ", string)
    return string.strip().split('.')

def process_labels(string):
    return [float(x) for x in string]


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
            if word.lower() not in word_dict:
                word_dict[word.lower()] = ''

    #unkown token for words not in GLOVE
    word_dict['unk'] = ''
    emb_list = []

    with open(glove_path, encoding="utf8") as f:
        for sents in f:
            word, emb = sents.split(' ',1)
            if word.lower() in word_dict:
                embeddings[word.lower()] = np.fromstring(emb, sep=' ')
                emb_list.append(np.fromstring(emb, sep=' '))
#                 np.array(list(map(float, emb.split())))
    emb_list = np.array(emb_list)
    avg_emb = np.mean(emb_list, axis = 0)
    embeddings['unk'] = avg_emb
    # print("\nEmbeddings of unk token: \n\n",embeddings['unk'])
    print("\nFound "+ str(len(embeddings)) + " words with Glove embeddings out of "+ str(len(word_dict)) + " total words in corpus.\n")
    return word_dict, embeddings


def get_batch_from_idx(config, word_emb, batch):
    doc_lens = np.array([len(x) for x in batch])
    max_len = np.max(doc_lens)
    embedded_docs = np.zeros((max_len, len(batch), config['embed_dim']))
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            if batch[i][j].lower() in word_emb:
                embedded_docs[j, i, :] = word_emb[batch[i][j].lower()]     # j,i and not i,j since we are working with batch_first = False for LSTMs
            else:
                embedded_docs[j, i, :] = word_emb['unk']
    return torch.from_numpy(embedded_docs).float(), torch.from_numpy(doc_lens)



#################################################
## Class to manage Reuters data pre-processing ##
#################################################

class Reuters(TabularDataset):
    TEXT = Field(sequential = True, batch_first=False, lower=True, use_vocab=True, tokenize=clean_string, include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=False, batch_first=False, preprocessing=process_labels)
    NUM_CLASSES = 90

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def get_dataset_splits(cls, data_dir, train=os.path.join('reuters_split', 'train.tsv'),
               validation=os.path.join('reuters_split', 'dev.tsv'),
               test=os.path.join('reuters_split', 'test.tsv'), **kwargs):

        return super(Reuters, cls).splits(
            data_dir, train=train, validation=validation, test=test,
            format='tsv', fields=[('label', cls.LABEL), ('text', cls.TEXT)])

    @classmethod
    def main_handler(cls, config, data_dir, shuffle=True):

        # Getting Data Splits: train, dev, test
        print("\n\n==>> Loading Data splits and tokenizing each document....")
        train, val, test = cls.get_dataset_splits(data_dir)

        # Build Vocabulary and obtain embeddings for each word in Vocabulary
        print("\n==>> Building Vocabulary and obtaining embeddings....")
        glove_embeds = torchtext.vocab.Vectors(name= config['glove_path'], max_vectors = int(2e5))
        cls.TEXT.build_vocab(train, val, test, vectors=glove_embeds)

        # Setting 'unk' token as the average of all other embeddings
        if config['model_name'] != 'han':
            cls.TEXT.vocab.vectors[cls.TEXT.vocab.stoi['<unk>']] = torch.mean(cls.TEXT.vocab.vectors, dim=0)

        # Getting iterators for each set
        print("\n==>> Preparing Iterators....")
        train_iter, val_iter, test_iter = BucketIterator.splits((train, val, test), batch_size=config['batch_size'], repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)
        return cls.TEXT, cls.LABEL, train_iter, val_iter, test_iter


class Reuters_HAN(Reuters):
    NESTING = Field(sequential = True, batch_first=True, lower=True, use_vocab=True, tokenize=clean_string)
    TEXT = NestedField(NESTING, tokenize=split_sents, include_lengths = True)


class Reuters_CNN(Reuters):
    TEXT = Field(sequential = True, batch_first=False, lower=True, use_vocab=True, tokenize=clean_string_stop_words_remove, include_lengths=True)




class ReutersBatchGenerator():
    def __init__(self, bucket_iterator, text_field = 'text', label_field = 'label'):
        self.bucket_iterator = bucket_iterator
        self.text_field = text_field
        self.label_field = label_field

    def __len__(self):
        return len(self.bucket_iterator)

    def __iter__(self):
        for batch in self.bucket_iterator:
            text = getattr(batch, self.text_field)
            label = getattr(batch, self.label_field)
            yield text, label