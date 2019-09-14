import sys, os, time, random, torchtext
import numpy as np
from data_utils import *
import warnings
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from torchtext.data import Field, Dataset, NestedField, TabularDataset, BucketIterator
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hacky trick to avoid the MAXSIZE python error
import csv
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/2)




def prepare_training(config, classes):

    print("="*80 + "\n\t\t\t\t Preparing Data\n" + "="*80)
    start = time.time()

    TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader = Reuters.iters(config, config['reuters_path'], shuffle=True)
    vocab_size = len(TEXT.vocab)
    config['vocab_size'] = vocab_size

    # print("\nExample document: \n", next(train_batch_loader.text))
    # print("\nLabels for the above document: \n", next(train_batch_loader.label))

    print("\n\nDATA STATISTICS:\n" + "-"*50)
    print("\nVocabulary size = ", vocab_size)
    print('No. of target classes = ', train_batch_loader.dataset.NUM_CLASSES)
    print('No. of train instances = ', len(train_batch_loader.dataset))
    print('No. of dev instances = ', len(dev_batch_loader.dataset))
    print('No. of test instances = ', len(test_batch_loader.dataset))

    # Custom wrapper over the iterators
    train_batch_loader = ReutersBatchGenerator(train_batch_loader)
    dev_batch_loader = ReutersBatchGenerator(dev_batch_loader)
    test_batch_loader = ReutersBatchGenerator(test_batch_loader)

    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    print("\n"+ "-"*50 + "\nTook  {:0>2} hours: {:0>2} mins: {:05.2f} secs  to Prepare Data\n".format(hours,minutes,seconds))
    # return train_x, train_y, val_x, val_y, test_data, test_y, vocab, embeddings
    return train_batch_loader, dev_batch_loader, test_batch_loader, TEXT, LABEL


def calc_elapsed_time(start, end):
    hours, rem = divmod(end-start, 3600)
    time_hours, time_rem = divmod(end, 3600)
    minutes, seconds = divmod(rem, 60)
    time_mins, _ = divmod(time_rem, 60)
    return int(hours), int(minutes), seconds



def evaluation_measures(config, preds, labels):
    f1 = f1_score(labels.to('cpu'), preds.to('cpu'), average = 'weighted')
    recall = recall_score(labels.to('cpu'), preds.to('cpu'), average = 'weighted')
    precision = precision_score(labels.to('cpu'), preds.to('cpu'), average = 'weighted')
    accuracy = accuracy_score(labels.to('cpu'), preds.to('cpu'), normalize= True)
    return f1, recall, precision, accuracy


def print_stats(config, epoch, train_acc, train_loss, train_f1, val_acc, val_f1, start):
    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    print(("Epoch: {}/{},     train_loss: {:.4f},    train_Acc = {:.4f},    train_f1 = {:.4f},     eval_acc = {:.4f},       eval_f1 = {:.4f}   |  Elapsed Time:  {:0>2}:{:0>2}:{:05.2f}"
                     .format(epoch, config['max_epoch'], train_loss, train_acc, train_f1, val_acc, val_f1, hours,minutes,seconds)))