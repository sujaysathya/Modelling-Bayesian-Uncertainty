import sys, os, time, random
import numpy as np
from data_utils import *

def prepare_training(config):

    print("="*80 + "\n\t\t\t\t Preparing Data\n" + "="*80)
    start = time.time()
    # Getting Data Splits: train, test
    print("\n\n==>> Getting Data splits..")
    train_docs, train_labels, test_docs, test_labels = get_data_splits()
    # print("\nTotal training docs (before val split): {} \nTotal test docs: {}\n".format(len(train_docs), len(test_docs)) + "-"*50)

    # TOKENIZING
    print("\n==>> Tokenizing each document...")
    print("\nTraining set:\n")
    train_data = tokenize(train_docs)
    print("\nTest set:\n")
    test_data = tokenize(test_docs)
    print("\n" + "-"*50)

    # Build Vocabulary and obtain embeddings for each word in Vocabulary
    print("\n==>> Building Vocabulary and obtaining embeddings...")
    vocab, embeddings = build_vocab(train_data, config['glove_path'])

    # Splitting training data into train-val split
    train_x, train_y, val_x, val_y = split_train_set(config, train_data, train_labels)
    print(val_x[0])
    print(val_y[0])
    print("-"*50 + "\nTotal training docs (after val split): {} \nTotal val docs: {} \nTotal test docs: {}\n".format(len(train_x), len(val_x), len(test_data)))

    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    print("\n"+ "-"*50 + "\nTook  {:0>2} hours: {:0>2} mins: {:05.2f} secs  to Prepare Data\n".format(hours,minutes,seconds))
    return (train_x, train_y), (val_x, val_y), (test_data, test_labels), vocab, embeddings


def calc_elapsed_time(start, end):
    hours, rem = divmod(end-start, 3600)
    time_hours, time_rem = divmod(end, 3600)
    minutes, seconds = divmod(rem, 60)
    time_mins, _ = divmod(time_rem, 60)
    return int(hours), int(minutes), seconds


def print_stats(epoch, train_loss, train_acc, val_acc, start):
    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    print(("Epoch: {}/{},    train_loss: {:.4f},  train_acc = {:.2f}   eval_acc = {:.2f}  | Elapsed Time:  {:0>2}:{:0>2}:{:05.2f}"
                     .format(epoch, config['max_epoch'], train_loss, train_acc, val_acc, hours,minutes,seconds)))