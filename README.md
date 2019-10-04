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

We then proceed to perform uncertainty analysis of BiLSTM-regularized model using [MC Dropout](https://arxiv.org/pdf/1506.02142.pdf). This allows us to model the posterior instead of the likelihood using a Burnoulli prior. We measure the uncertainties by means of sampling 200 forward passes and reporting the mean and standard deviation of the prediciton scores to observe the model's behavior around the decision boundary.

## Dataset

[ApteMod](https://www.kaggle.com/nltkdata/reuters/version/2) is a collection of 10,788 documents from the Reuters financial newswire service, partitioned into a training set with 7769 documents and a test set with 3019 documents. In the ApteMod corpus, each document belongs to one or more categories. There are 90 categories in the corpus.

## Setting up the environment

1. Run the command `conda create -n name_of_env_here python=3.7.3` to create a virtual environment for the project.
2. Then run `conda activate name_of_env_just_created` to enter that virtual environment you just created.
3. Install all the dependencies by running the command `pip install -r requirements.txt` after cloning this repository.

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
2. 
3. 
4. 
5. 
6. 

#### Training parameters

7.
8.
9.
10.


### Code Overview


