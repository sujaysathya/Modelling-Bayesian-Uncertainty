# Modelling Bayesian Uncertainty in Neural Networks for Multi-label Document Classification
Project for the course Deep Learning for Natural Language Processing

## Task Description

In this task we classify the [ApteMod](https://www.kaggle.com/nltkdata/reuters/version/2) Reuters documents. It is a multi-class, multi-label classificaiton task with a granularity of 90 classes.

We propose to compare performance of [HAN](https://www.aclweb.org/anthology/N16-1174.pdf), [Kim-CNN](https://arxiv.org/pdf/1408.5882.pdf) and [BiLSTM-regularized](https://www.aclweb.org/anthology/N19-1408.pdf)  at this task of multi label document classification. Our baseline models are vanilla BiLSTM and BiLSTM max-pooling-over-time variant. The results are listed below:

|                 | Dev (F1) | Test (F1) | Dev (accuracy) | Test (accuracy) | No. of parameters |
|:---------------:|:--------:|:---------:|:--------------:|:---------------:|:-----------------:|
|      BiLSTM     |   78.32  |   75.55   |      68.13     |      67.41      |     1,665,024     |
| BiLSTM(maxpool) |   84.47  |   72.79   |      75.46     |      72.79      |     1,665,024     |
|     Kim-CNN     |   85.8   |   84.2    |      77.05     |      76.65      |      462,336      |
|      HAN        |   84.67  |   84.2    |      79.05     |      68.48      |      390,800      |
|  BiLSTM(reg)    |   88.14  |   86.43   |      80.82     |      76.68      |     1,665,024     |

We then proceed to perform uncertainty analysis of BiLSTM-regularized model using [MC Dropout](https://arxiv.org/pdf/1506.02142.pdf). This allows us to model the posterior instead of the likelihood using a Burnoulli prior. We measure the uncertainties by means of sampling 200 forward passes and reporting the mean and standard deviation of the prediciton scores to observe the model's behavior around the decision boundary. Uncertainty estimate of a subset of classes for one document of the test set is shown below:

![Uncertainty - decision boundary](https://github.com/shaanchandra/DeepLearningForNLP/blob/master/uncert_dec_boundary.png)

We can see that even though the mean is below the 0.5 mark and hence in likelihood estimation the model will classify it correctly, the uncertainty estimate shows that the model is still not sure about this prediction. We can see that in some of the forward passes, the score was actually greater than 0.5 and hence misclassified.

## Dataset

[ApteMod](https://www.kaggle.com/nltkdata/reuters/version/2) is a collection of 10,788 documents from the Reuters financial newswire service, partitioned into a training set with 7769 documents and a test set with 3019 documents. In the ApteMod corpus, each document belongs to one or more categories. There are 90 categories in the corpus.

Note, you do not have to download the dataset again. You can find the standard train, dev and test splits of the dataset in the `data/reuters_split` folder of this repository.

## Setting up the environment

1. Run the command `conda create -n name_of_env_here python=3.7.3` to create a virtual environment for the project.
2. Then run `conda activate name_of_env_just_created` to enter that virtual environment you just created.
3. Install all the dependencies by running the command `pip install -r requirements.txt` after cloning this repository.

***NOTE:*** before running Step3 above, you might have to install `torch` package as per the command [here](https://pytorch.org/get-started/locally/) based on your system. You can then run the command in Step3.

It is advisable to maintain the hierarchy of folders as present here. However, if you change the structure of code make sure to be consistent with path arguments too.

## Code

### Run a demo

To run the code with default parameters (BiLSTM-regularized on reuters dataset with the parameters setting dexcribed in the paper):

1. You will have to download the GloVE embeddings ([840b300d](https://nlp.stanford.edu/projects/glove/)), unzip it and palce the `.txt` file at the location `data/glove/glove.840B.300d.txt`.
2. Switch directories using `cd code` and then run `python3 train.py`. 
3. Note, this will create necessary the model saving checkpoints, run a check on all other necessary paths and attribute values provided and then run the data pre-processing steps, train the network, store the best model, load the best model and evaluate it on the test set.
4. The progress of training and data pre-processing will be visible on the screen.

Except for *Kim-CNN*, it is recommended to run the models on a GPU.

### Details of parser arguments

You can control all the necessary settings through the parser arguments of `train.py`. Below is the list of all arguments, their default values and details of purpose: 

#### Required Paths

1. ***data_path***(str): path to dataset folder that contains the folders to reuters or imdb (raw data)
2. ***glove_path***(str): path for Glove embeddings (850B, 300D)
3. ***model_checkpoint_path***(str): Directory for saving trained model checkpoints
4. ***vis_path***(str): Directory for saving tensorboard checkpoints
5. ***model_save_name***(str): saved model name

#### Training parameters

1. ***data_name***(str): dataset name: reuters / imdb
2. ***model_name***(str): model name: bilstm / bilstm_pool / bilstm_reg / han / cnn
3. ***lr***(float): default = 0.01, Learning rate for training
4. ***batch_size***(int): default = 32, batch size for training
5. ***embed_dim***(int): default = 300, dimension of word embeddings used(GLove)
6. ***lstm_dim***(int); default = 256, dimen of hidden unit of LSTM/BiLSTM networks
7. ***word_gru_dim***(int): default = 50, dimen of hidden unit of word-level attn GRU units of HAN
8. ***sent_gru_dim***(int): default = 50, dimen of hidden unit of sentence-level attn GRU units of HAN
9. ***fc_dim***(int): default = 128, dimen of FC layer
10. ***n_classes***(int): default = 90, number of classes
11. ***optimizer***(str): default = 'Adam', Optimizer to use for training
12. ***weight_decay***(float): default = 0, weight decay for optimizer
13. ***momentum***(float): default = 0.8, Momentum for optimizer
14. ***max_epoch***(int): default = 50, Max epochs to train for
15. ***val_split***(int): default = 0.1, Ratio of training data to be split into validation set
16. ***lr_decay_step***(float): default = 2000, Number of steps after which learning rate should be decreased
17. ***lr_decay_factor***(float): default = 0.2, Decay of learning rate of the optimizer
18. ***lr_cut_off***(float): default = 1e-7, Lr lower bound to stop training
19. ***beta_ema***(float): default = 0.99, Temporal Averaging smoothing co-efficient
20. ***wdrop***(float): default = 0.2, Regularization - weight dropout
21. ***embed_drop***(float): default = 0.1, Regularization - embedding dropout
22. ***dropout***(float): default = 0.5, Regularization - dropout in LSTM cells
23. ***kernel-num***(int): default=100, number of each kind of kernel
24. ***kernel-sizes***(str): default='3,4,5', comma-separated kernel size to use for convolution
25. ***seed***(int): default=2424, set seed for reproducability
26. ***bayesian_mode***(bool): default=True, To run the model in Bayesian Uncertainty analysis mode or not


### Code Overview


