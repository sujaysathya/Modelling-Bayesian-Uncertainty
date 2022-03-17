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
    # decrease the maxInt value by factor 2 as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/2)


# For printing cleaner numpy arrays
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def prepare_training(config):

    print("="*100 + "\n\t\t\t\t\t Preparing Data\n" + "="*100)
    start = time.time()

    if config['data_name'] == 'reuters':
        if config['model_name'] == 'han':
            TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = Reuters_HAN.main_handler(config, config['data_path'], shuffle=True)
        else:
            TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = Reuters.main_handler(config, config['data_path'], shuffle=True)
    
    elif config['data_name'] == 'cmu':
        TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = CMU.main_handler(config, config['data_path'], shuffle=True)
        # print(list(torch.utils.data.DataLoader(CMU)))
    else:
        if config['model_name'] == 'han':
            TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = IMDB_HAN.main_handler(config, config['data_path'], shuffle=True)
        else:
            TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = IMDB.main_handler(config, config['data_path'], shuffle=True)

    vocab_size = len(TEXT.vocab)
    config['vocab_size'] = vocab_size

    # Creating class distributions for each set from the torchtext LABELs field
    print("\n\n==>> Creating class distributions...")
    train_sorted, val_sorted, test_sorted, train_sorted_idx, val_sorted_idx, test_sorted_idx = get_distributions(config, train_split, val_split, test_split)


    print("\n\n" + "-"*50 + "\nDATA STATISTICS:\n" + "-"*50)
    print("\nVocabulary size = ", vocab_size)
    print('No. of target classes = ', train_batch_loader.dataset.NUM_CLASSES)
    print('No. of train instances = ', len(train_batch_loader.dataset))
    print('No. of dev instances = ', len(dev_batch_loader.dataset))
    print('No. of test instances = ', len(test_batch_loader.dataset))
    print("\nTop 10 training set classes by ratio:    Classes  {}   with % of  {}".format(str(train_sorted_idx[:5]+1), train_sorted[:5]))
    print("Top 10 validation set classes by ratio:    Classes  {}   with % of  {}".format(str(val_sorted_idx[:5]+1), val_sorted[:5]))
    print("Top 10 test set classes by ratio:    Classes  {}   with % of  {}".format(str(test_sorted_idx[:5]+1), test_sorted[:5]))
    print("\nTop 10 RARE training set classes by ratio:    Classes  {}   with % of  {}".format(str(train_sorted_idx[-5:]+1), train_sorted[-5:]))
    print("Top 10 RARE validation set classes by ratio:    Classes  {}   with % of  {}".format(str(val_sorted_idx[-5:]+1), val_sorted[-5:]))
    print("Top 10 RARE test set classes by ratio:    Classes  {}   with % of  {}".format(str(test_sorted_idx[-5:]+1), test_sorted[-5:]))
    zero = sorted([train_sorted_idx[i] for i,x in enumerate(train_sorted) if x<1e-3])
    print("\nClasses with zero support in training set = {}".format(zero))
    zero = sorted([val_sorted_idx[i] for i,x in enumerate(val_sorted) if x < 1e-3])
    print("Classes with zero support in validation set = {}".format(zero))
    zero = sorted([test_sorted_idx[i] for i,x in enumerate(test_sorted) if x < 1e-3])
    print("Classes with zero support in test set = {}".format(zero))


    # Custom wrapper over the iterators
    # train_batch_loader = ReutersBatchGenerator(train_batch_loader)
    # dev_batch_loader = ReutersBatchGenerator(dev_batch_loader)
    # test_batch_loader = ReutersBatchGenerator(test_batch_loader)


    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    print("\n"+ "-"*50 + "\nTook  {:0>2} hours: {:0>2} mins: {:05.2f} secs  to Prepare Data\n".format(hours,minutes,seconds))
    return train_batch_loader, dev_batch_loader, test_batch_loader, TEXT, LABEL


def calc_elapsed_time(start, end):
    hours, rem = divmod(end-start, 3600)
    time_hours, time_rem = divmod(end, 3600)
    minutes, seconds = divmod(rem, 60)
    time_mins, _ = divmod(time_rem, 60)
    return int(hours), int(minutes), seconds



def get_distributions(config, train_split, val_split, test_split):
    classes = 90 if config['data_name'] == 'reuters' else 227

    train_distrib, val_distrib, test_distrib = np.zeros((len(train_split), classes)), np.zeros((len(val_split), classes)), np.zeros((len(test_split), classes))
    sets = ['train_split', 'val_split', 'test_split']
    dists = ['train_distrib', 'val_distrib', 'test_distrib']
    sorts = ['train_sorted', 'val_sorted', 'test_sorted']
    sorted_idx = ['train_sorted_idx', 'val_sorted_idx', 'test_sorted_idx']
    train_sorted, val_sorted, test_sorted = [], [], []
    train_sorted_idx, val_sorted_idx, test_sorted_idx = [], [], []

    for s in range(len(sets)):
        st = eval(sets[s])
        # print("ST: "+str(st))
        # label_sizes = []
        # for i in range(len(st)):
        #     label_sizes.append(len(st[i].label))
        #     if i < 5:
        #         print(st[i].label)
        #         print("ST TEXT: " + str(st[i].text))
        #     # print("this is what you want")
        #     # print(st[i].label)
        # print("Max label size: "+str(max(label_sizes)))
        # print("Max label size: "+str(min(label_sizes)))


        for i in range(len(st)):
            eval(dists[s])[i, :] = st[i].label
            # print("this is what you want")
            # print(st[i].label)

    for i in range(len(sets)):
        temp = []
        # print("\nClass distributions for  {}\n".format(sets[i]) + "-"*40)
        all_classes_sum = np.sum(eval(dists[i]), axis =0)
        for j in range(classes):
            this_class_ratio = all_classes_sum[j]/len(eval(sets[i]))
            # print("Class {0}:   {1:.4f} %".format(j+1, this_class_ratio*100))
            temp.append(this_class_ratio)
        eval(sorts[i]).append(-np.sort(-np.array(temp)))
        eval(sorted_idx[i]).append(np.argsort(-np.array(temp)))
    return train_sorted[0]*100, val_sorted[0]*100, test_sorted[0]*100, train_sorted_idx[0], val_sorted_idx[0], test_sorted_idx[0]



def evaluation_measures(config, preds, labels):
    f1 = f1_score(labels, preds, average = 'micro')
    recall = recall_score(labels, preds, average = 'micro')
    precision = precision_score(labels, preds, average = 'micro')
    accuracy = accuracy_score(labels, preds)
    return f1, recall, precision, accuracy


def print_stats(config, epoch, train_acc, train_loss, train_f1, val_acc, val_f1, start):
    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    print(("Epoch: {}/{},     train_loss: {:.4f},    train_Acc = {:.4f},    train_f1 = {:.4f},     eval_acc = {:.4f},       eval_f1 = {:.4f}   |  Elapsed Time:  {:0>2}:{:0>2}:{:05.2f}"
                     .format(epoch, config['max_epoch'], train_loss, train_acc, train_f1, val_acc, val_f1, hours,minutes,seconds)))