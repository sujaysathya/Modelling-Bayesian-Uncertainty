import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Baseline(nn.Module):
    def __init__(self, config):
        super(Baseline, self).__init()

    def forward(self, input):

        return None




class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()


    def forward(Self, input):

        return None




class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init()

    def forward(self, input):

        return None