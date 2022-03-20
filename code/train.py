from utils import *
from models import *
from data_utils import *
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.nn as nn
from torchtext import datasets
from torchtext.data import Field, BucketIterator
from torch.autograd import Variable
import nltk
from nltk import word_tokenize
from torch.utils.tensorboard import SummaryWriter
import torchtext
import torch
import numpy as np
import argparse
import time
import datetime
import shutil
import pprint
import sys
import os
import glob
import argparse
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")


# from tensorboardX import SummaryWriter
nltk.download('punkt')


def eval_network(model, test=False):
    model.eval()
    if config['model_name'] == 'bilstm_reg':
        if hasattr(model.encoder, 'beta_ema') and model.encoder.beta_ema > 0:
            # Temporal Averaging
            old_params = model.encoder.get_params()
            model.encoder.load_ema_params()

    eval_precision, eval_recall, eval_f1, eval_accuracy = [], [], [], []
    batch_loader = dev_loader if not test else test_loader
    reps = 500 if (config['bayesian_mode'] and test) else 1
    class_means, class_std = torch.zeros(32, 90), torch.zeros(32, 90)

    with torch.no_grad():
        for iters, batch in enumerate(batch_loader):
            rep_preds = []
            for i in range(reps):
                preds = model(batch.text[0].to(device),
                              batch.text[1].to(device))
                # preds = (preds>0.5).type(torch.FloatTensor)
                
                preds_rounded = F.sigmoid(preds).round().long()
                true_labels = batch.label
                # if preds.shape[0] == config['batch_size']:
                rep_preds.append(F.sigmoid(preds))

                f1, recall, precision, accuracy = evaluation_measures(config, np.array(
                    preds_rounded.cpu().detach().numpy()), np.array(true_labels.cpu().detach().numpy()))

                eval_f1.append(f1)
                eval_precision.append(precision)
                eval_recall.append(recall)
                eval_accuracy.append(accuracy)

            rep_preds = torch.stack(rep_preds)
            batch_means = torch.mean(rep_preds, dim=0)
            batch_std = torch.std(rep_preds, dim=0)
            if iters == 0:
                class_means = batch_means
                class_std = batch_std
            else:
                class_means = torch.cat([class_means, batch_means], dim=0)
                class_std = torch.cat([class_std, batch_std], dim=0)
        
        # Averaging out the results
        eval_precision = sum(eval_precision)/len(eval_precision)
        eval_recall = sum(eval_recall)/len(eval_recall)
        eval_f1 = sum(eval_f1)/len(eval_f1)
        eval_accuracy = sum(eval_accuracy)/len(eval_accuracy)

        if hasattr(model.encoder, 'beta_ema') and model.encoder.beta_ema > 0:
            # Temporal averaging
            model.encoder.load_params(old_params)
    return eval_f1, eval_precision, eval_recall, eval_accuracy, class_means, class_std


def train_network():
    print("_"*100 + "\n\t\t\t\t\t Training Network\n" + "_"*100)

    # Seeds for reproduceable runs
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the model, optimizer and loss function
    model = Document_Classifier(
        config, pre_trained_embeds=TEXT.vocab.vectors).to(device)
    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(
        ), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])

    criterion = nn.CrossEntropyLoss()
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size= config["lr_decay_step"], gamma= config["lr_decay_factor"])

    # Load the checkpoint to resume training if found
    model_file = os.path.join(config['model_checkpoint_path'], config['data_name'],
                              config['model_name'], str(config['seed']), config['model_save_name'])

    if os.path.isfile(model_file):
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("Resuming training from epoch {} with loaded model and optimizer".format(
            start_epoch))
        print("Using the model defined below:")
        print(model)

    else:
        start_epoch = 1
        print("\nNo Checkpoints found for the chosen model to reusme training.Training the  ''{}''  model from scratch".format(config['model_name']))
        print("Using the model defined below:")
        print(model)

    start = time.time()
    best_val_acc = 0
    best_val_f1 = 0
    prev_val_acc = 0
    prev_val_f1 = 0
    total_iters = 0
    train_loss = []
    MEANS, STD = [], []
    terminate_training = False

    print("Beginning training at:  {}".format(datetime.datetime.now()))
    # for epoch in range(start_epoch, config['max_epoch']+1):
    for epoch in range(50):
        train_f1_score, train_recall_score, train_precision_score, train_accuracy_score = [], [], [], []
        model.train()

        for iters, batch in enumerate(train_loader):
            model.train()
            # lr_scheduler.step()
            preds = model(batch.text[0].to(device), batch.text[1].to(device))
            loss = F.binary_cross_entropy_with_logits(preds, batch.label.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            if config['model_name'] == 'bilstm_reg':
                if hasattr(model.encoder, 'beta_ema') and model.encoder.beta_ema > 0:
                    # Temporal averaging
                    model.encoder.update_ema()

            # preds = (preds>0.5).type(torch.FloatTensor)
            
            preds_rounded = F.sigmoid(preds).round().long()
            true_labels = batch.label
            # print(true_labels)
            # print(preds_rounded)
            # print(preds)
            # print("-"*40)
            train_f1, train_recall, train_precision, train_accuracy = evaluation_measures(config, np.array(
                preds_rounded.cpu().detach().numpy()), np.array(true_labels.cpu().detach().numpy()))

            train_f1_score.append(train_f1)
            train_accuracy_score.append(train_accuracy)
            train_recall_score.append(train_recall)
            train_precision_score.append(train_precision)
            train_loss.append(loss.detach().item())
            if iters % 100 == 0:
                writer.add_scalar('Train/loss', sum(train_loss) /
                                  len(train_loss), ((iters+1) + total_iters))
                writer.add_scalar('Train/precision', sum(train_precision_score) /
                                  len(train_precision_score), ((iters+1)+total_iters))
                writer.add_scalar('Train/recall', sum(train_recall_score) /
                                  len(train_recall_score), ((iters+1)+total_iters))
                writer.add_scalar('Train/f1', sum(train_f1_score) /
                                  len(train_f1_score), ((iters+1)+total_iters))
                writer.add_scalar('Train/accuracy', sum(train_accuracy_score) /
                                  len(train_accuracy_score), ((iters+1)+total_iters))

                for name, param in model.encoder.named_parameters():
                    if not param.requires_grad:
                        continue
                    writer.add_histogram(
                        'iters/'+name, param.data.view(-1), global_step=((iters+1)+total_iters))
                    writer.add_histogram(
                        'grads/' + name, param.grad.data.view(-1), global_step=((iters+1) + total_iters))

        total_iters += iters

        # Evaluate on test set
        eval_f1, eval_precision, eval_recall, eval_accuracy, class_means, class_std = eval_network(model)

        if config['bayesian_mode']:
            MEANS.append(class_means)
            STD.append(class_std)
            torch.save({
                'class_means': MEANS,
                'class_std': STD,
            }, os.path.join(config['model_checkpoint_path'], config['data_name'], config['model_name'], str(config['seed']), 'bayesian_uncertainties_val.pt'))

        # print stats
        print_stats(config, epoch, sum(train_accuracy_score)/len(train_accuracy_score), sum(train_loss) /
                    len(train_loss), sum(train_f1_score)/len(train_f1_score), eval_accuracy, eval_f1, start)

        writer.add_scalar('Validation/f1', eval_f1, epoch)
        writer.add_scalar('Validation/recall', eval_recall, epoch)
        writer.add_scalar('Validation/precision', eval_precision, epoch)
        writer.add_scalar('Validation/accuracy', eval_accuracy, epoch)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            writer.add_histogram(
                'epochs/' + name, param.data.view(-1), global_step=epoch)

        # Save model checkpoints for best model
        if eval_f1 > best_val_f1:
            print("New High Score! Saving model...\n")
            best_val_f1 = eval_f1
            best_val_acc = eval_accuracy
            # Save the state and the vocabulary
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(config['model_checkpoint_path'], config['data_name'], config['model_name'], str(config['seed']), config['model_save_name']))

        # If validation f1 score does not improve for 10 epochs, divide the learning rate by 5 and
        # if learning rate falls below given threshold, then terminate training
        epoch_count = 1
        if eval_f1 <= prev_val_f1:
            for param_group in optimizer.param_groups:
                if param_group['lr'] < config['lr_cut_off']:
                    terminate_training = True
                    break
                elif epoch_count == 0:
                    param_group['lr'] /= 5
                    print("Learning rate changed to :  {}\n".format(param_group['lr']))
                epoch_count = (epoch_count+1)%10

        prev_val_f1 = eval_f1
        if terminate_training:
            break

    # Termination message
    if terminate_training:
        print("-"*100 +"Training terminated because the learning rate fell below:  {}" .format(config['lr_cut_off']))
    else:
        print("-"*100 + "\nMaximum epochs reached. Finished training !!")

    print("-"*50 + "\n\t\tEvaluating on test set\n" + "-"*50)
    model_file = os.path.join(config['model_checkpoint_path'], config['data_name'],
                              config['model_name'], str(config['seed']), config['model_save_name'])
    if os.path.isfile(model_file):
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("No Saved model state_dict found for the chosen model!!! \nAborting evaluation.".format(
            config['model_name']))
    test_f1, test_precision, test_recall, test_accuracy, class_means, class_std = eval_network(
        model, test=True)
    if config['bayesian_mode']:
        torch.save({
            'class_means': class_means,
            'class_std': class_std,
        }, os.path.join(config['model_checkpoint_path'], config['data_name'], config['model_name'], str(config['seed']), 'bayesian_uncertainties_test.pt'))
    print("\nTest precision of best model = {:.2f}".format(test_precision*100))
    print("\nTest recall of best model = {:.2f}".format(test_recall*100))
    print("\nTest f1 of best model = {:.2f}".format(test_f1*100))
    print("\nTest accuracy of best model = {:.2f}".format(test_accuracy*100))

    writer.close()
    return best_val_f1, best_val_acc, test_f1, test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required Paths
    parser.add_argument('--data_path', type=str, default='../data',
                        help='path to dataset folder that contains the folders to reuters')
    parser.add_argument('--glove_path', type=str, default='../data/glove/glove.840B.300d.txt',
                        help='path for Glove embeddings (850B, 300D)')
    parser.add_argument('--model_checkpoint_path', type=str, default='./model_checkpoints',
                        help='Directory for saving trained model checkpoints')
    parser.add_argument('--vis_path', type=str, default='./vis_checkpoints',
                        help='Directory for saving tensorboard checkpoints')
    parser.add_argument("--model_save_name", type=str, default='best_model.pt',
                        help='saved model name')

    # Training Params
    parser.add_argument('--data_name', type=str, default='reuters',
                        help='dataset name: reuters / cmu')
    parser.add_argument('--model_name', type=str, default='cnn',
                        help='model name: bilstm / bilstm_pool / bilstm_reg / cnn')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training"')
    parser.add_argument('--embed_dim', type=int, default=300,
                        help='dimension of word embeddings used(GLove)"')
    parser.add_argument('--lstm_dim', type=int, default=256,
                        help='dimen of hidden unit of LSTM/BiLSTM networks"')
    parser.add_argument('--word_gru_dim', type=int, default=50,
                        help='dimen of hidden unit of word-level attn GRU units of HAN"')
    parser.add_argument('--sent_gru_dim', type=int, default=50,
                        help='dimen of hidden unit of sentence-level attn GRU units of HAN"')
    parser.add_argument('--fc_dim', type=int, default=128,
                        help='dimen of FC layer"')
    parser.add_argument('--n_classes', type=int, default=20,
                        help='number of classes"')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer to use for training')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.8,
                        help='Momentum for optimizer')
    parser.add_argument('--max_epoch', type=int, default=50,
                        help='Max epochs to train for')
    parser.add_argument('--val_split', type=int, default=0.1,
                        help='Ratio of training data to be split into validation set')
    parser.add_argument('--lr_decay_step', type=float, default=2000,
                        help='Number of steps after which learning rate should be decreased')
    parser.add_argument('--lr_decay_factor', type=float, default=0.2,
                        help='Decay of learning rate of the optimizer')
    parser.add_argument('--lr_cut_off', type=float, default=1e-7,
                        help='Lr lower bound to stop training')
    parser.add_argument('--beta_ema', type=float, default=0.99,
                        help='Temporal Averaging smoothing co-efficient')
    parser.add_argument('--wdrop', type=float, default=0.2,
                        help='Regularization - weight dropout')
    parser.add_argument('--embed_drop', type=float, default=0.1,
                        help='Regularization - embedding dropout')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Regularization - dropout in LSTM cells')
    parser.add_argument('--kernel-num', type=int, default=100,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel-sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--seed', type=int, default=2424,
                        help='set seed for reproducability')
    parser.add_argument('--bayesian_mode', type=bool, default=True,
                        help='To run the model in Bayesian Uncertainty analysis mode or not')

    args, unparsed = parser.parse_known_args()
    config = args.__dict__

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    # Check all provided paths:
    model_path = os.path.join(config['model_checkpoint_path'],
                              config['data_name'], config['model_name'], str(config['seed']))
    vis_path = os.path.join(
        config['vis_path'], config['data_name'], config['model_name'])
    if not os.path.exists(config['data_path']):
        raise ValueError("[!] ERROR: Dataset path does not exist")
    else:
        print("Data path checked")
    if not os.path.exists(config['glove_path']):
        raise ValueError("[!] ERROR: GLOVE Embeddings path does not exist")
    else:
        print("GLOVE embeddings path checked")
    if not os.path.exists(model_path):
        print("Creating checkpoint path for saved models at:  {}\n".format(model_path))
        os.makedirs(model_path)
    else:
        print("Model save path checked..")
    if config['model_name'] not in ['bilstm', 'bilstm_pool', 'bilstm_reg', 'han', 'cnn']:
        raise ValueError("[!] ERROR:  model_name is incorrect. Choose one of - bilstm / bilstm_pool / bilstm_reg / cnn")
    else:
        print("Model name checked")
    if not os.path.exists(vis_path):
        print("Creating checkpoint path for Tensorboard visualizations at:  {}\n".format(
            vis_path))
        os.makedirs(vis_path)
    else:
        print("Tensorbaord Visualization path checked")
        print("Cleaning Visualization path of older tensorboard files.\n")
        shutil.rmtree(vis_path)
    config['n_classes'] = 20

    # Prepare the datasets and iterator for training and evaluation
    train_loader, dev_loader, test_loader, TEXT, LABEL = prepare_training(
        config)
    vocab = TEXT.vocab

    # Print args
    print("x"*50 + "\nRunning training with the following parameters:")
    for key, value in config.items():
        print(key + ' : ' + str(value))
    print("x"*50)

    # Prepare the tensorboard writer
    writer = SummaryWriter(os.path.join(args.vis_path, config['model_name']))

    try:
        train_network()
    except KeyboardInterrupt:
        print("Keyboard interrupt by user detected\nClosing the tensorboard writer!")
        writer.close()