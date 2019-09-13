import sys, os, time, random
import numpy as np
from data_utils import *
import warnings
from sklearn.metrics import f1_score, recall_score, precision_score
warnings.filterwarnings("ignore")

def prepare_training(config, classes):

    print("="*80 + "\n\t\t\t\t Preparing Data\n" + "="*80)
    start = time.time()
    # Getting Data Splits: train, test
    print("\n\n==>> Getting Data splits....")
    train_docs, train_labels, test_docs, test_labels = get_data_splits()

    # TOKENIZING
    print("\n==>> Tokenizing each document....")
    print("\nTraining set:\n")
    train_data = tokenize(train_docs)
    print("\nTest set:\n")
    test_data = tokenize(test_docs)
    print("\n" + "-"*50)

    # Build Vocabulary and obtain embeddings for each word in Vocabulary
    print("\n==>> Building Vocabulary and obtaining embeddings....")
    vocab, embeddings = build_vocab(train_data, config['glove_path'])

    # Splitting training data into train-val split
    train_x, train_y, val_x, val_y = split_train_set(config, train_data, train_labels)
    print("-"*50 + "\nTotal training docs (after val split): {} \nTotal val docs: {} \nTotal test docs: {}\n".format(len(train_x), len(val_x), len(test_data)))
    print("\nExample train doc: \n{}\n".format(train_x[0]))
    print("\nExample train doc's label: \n{}\n".format(train_y[0]))

    # # Testing labels integrity
    # print("\n==>> Testing labels....")
    # testing_labels(train_y, val_y, test_labels)

    # Binarizing the labels for NN-training
    print("\n==>> Binarizing labels for trianing....")
    train_y, val_y, test_y = binarize_labels(classes, train_y, val_y, test_labels)

    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    print("\n"+ "-"*50 + "\nTook  {:0>2} hours: {:0>2} mins: {:05.2f} secs  to Prepare Data\n".format(hours,minutes,seconds))
    return train_x, train_y, val_x, val_y, test_data, test_y, vocab, embeddings


def calc_elapsed_time(start, end):
    hours, rem = divmod(end-start, 3600)
    time_hours, time_rem = divmod(end, 3600)
    minutes, seconds = divmod(rem, 60)
    time_mins, _ = divmod(time_rem, 60)
    return int(hours), int(minutes), seconds



def evaluation_measures(config, preds, labels):
    # TP += ((preds == labels).float() * (preds == 1).float()).sum(dim=(0,1)).cpu().data.numpy()
    # FP += ((preds != labels).float() * (preds == 1).float()).sum(dim=(0,1)).cpu().data.numpy()
    # TN += ((preds == labels).float() * (preds == 0).float()).sum(dim=(0,1)).cpu().data.numpy()
    # FN += ((preds != labels).float() * (preds == 0).float()).sum(dim=(0,1)).cpu().data.numpy()

    # accuracy_per_label = (TP + TN) / (TP + FP + TN + FN + 1e-10)
    # balanced_accuracy_per_label = (TP/(TP+FN+1e-10) + TN/(TN+FP+1e-10)) / 2.0
    # precision_per_label = TP / (TP + FP + 1e-10)
    # recall_per_label = TP / (TP + FN + 1e-10)
    # f1_per_label = 2 * precision_per_label * recall_per_label / (1e-5 + precision_per_label + recall_per_label)
    # labels_per_class = TP + FP + TN + FN
    # print(accuracy_per_label)
    f1 = f1_score(labels, preds, average = 'weighted')
    recall = recall_score(labels, preds, average = 'weighted')
    precision = precision_score(labels, preds, average = 'weighted')
    # return TP, FP, TN, FN, torch.mean(torch.tensor(accuracy_per_label)), torch.mean(torch.tensor(balanced_accuracy_per_label)), torch.mean(torch.tensor(precision_per_label)), torch.mean(torch.tensor(recall_per_label)), torch.mean(torch.tensor(f1_per_label))
    return f1, recall, precision

def print_stats(config, epoch, train_loss, train_f1, val_f1, start):
    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    print(("Epoch: {}/{},      train_loss: {:.4f},    train_f1 = {:.4f},      eval_f1 = {:.4f}   |  Elapsed Time:  {:0>2}:{:0>2}:{:05.2f}"
                     .format(epoch, config['max_epoch'], train_loss, train_f1, val_f1, hours,minutes,seconds)))