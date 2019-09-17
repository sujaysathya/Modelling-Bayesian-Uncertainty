import torch
import numpy as np
from torch.nn import Parameter
from torchnlp.nn import WeightDrop
from functools import wraps
from copy import deepcopy
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


######################
## Helper Functions ##
######################

"""
The following 2 implementations are taken from the implementation of LSTM-reg in the HEDWIG framework
(https://github.com/castorini/hedwig/tree/master/models/reg_lstm)

"""

def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
      mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
      masked_embed_weight = mask * embed.weight
    else:
      masked_embed_weight = embed.weight
    if scale:
      masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight
      
    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
      
    X = torch.nn.functional.embedding(words, masked_embed_weight, padding_idx, embed.max_norm, embed.norm_type, embed.scale_grad_by_freq, embed.sparse)
    return X



<<<<<<< HEAD
class WeightDrop_manual(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()
=======
# class WeightDrop_manual(torch.nn.Module):
#     def __init__(self, module, weights, dropout=0, variational=False):
#         super().__init__()
#         self.module = module
#         self.weights = weights
#         self.dropout = dropout
#         self.variational = variational
#         self._setup()
>>>>>>> f712068342f99b8f512a3f8bff61991c53da4ece

#     def null_function(*args, **kwargs):
#         # We need to replace flatten_parameters with a nothing function
#         return

#     def _setup(self):
#         # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
#         if issubclass(type(self.module), torch.nn.RNNBase):
#             self.module.flatten_parameters = self.null_function

#         for name_w in self.weights:
#             print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
#             w = getattr(self.module, name_w)
#             del self.module._parameters[name_w]
#             self.module.register_parameter(name_w + '_raw', Parameter(w.data))

<<<<<<< HEAD
    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda:
                    mask = mask.cuda()
                    mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                    w = torch.nn.Parameter(mask.expand_as(raw_w) * raw_w)
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training).to(device)
            setattr(self.module, name_w, w)
=======
#     def _setweights(self):
#         for name_w in self.weights:
#             raw_w = getattr(self.module, name_w + '_raw')
#             w = None
#             if self.variational:
#                 mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
#                 if raw_w.is_cuda:
#                     mask = mask.cuda()
#                     mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
#                     w = torch.nn.Parameter(mask.expand_as(raw_w) * raw_w)
#             else:
#                 w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training).to(device)
#             setattr(self.module, name_w, w)
>>>>>>> f712068342f99b8f512a3f8bff61991c53da4ece

#     def forward(self, *args):
#         self._setweights()
#         return self.module.forward(*args)




###################
## Model Classes ##
###################

"""
Main class that controls training and calling of other classes based on corresponding model_name

"""
class Doc_Classifier(nn.Module):
    def __init__(self, config, pre_trained_embeds = None):
        super(Doc_Classifier, self).__init__()

        self.lstm_dim = config['lstm_dim']
        self.model_name = config['model_name']
        self.fc_dim = config['fc_dim']
        self.num_classes = config['n_classes']
        self.vocab_size = config['vocab_size']
        self.embed_dim = config['embed_dim']
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embedding.weight.data.copy_(pre_trained_embeds)
        self.embedding.requires_grad = False

        if self.model_name == 'bilstm':
            self.encoder = BiLSTM(config)
        elif self.model_name == 'bilstm_pool':
            self.encoder = BiLSTM(config, max_pool = True)
        elif self.model_name == 'bilstm_reg':
            self.encoder = BiLSTM_reg(config)
        elif self.model_name == 'han':
            self.encoder = HAN(config)
        elif self.model_name == 'cnn':
            self.encoder = Kim_CNN(config)


        if self.model_name == 'bilstm_reg':
            self.classifier = nn.Sequential(nn.Dropout(config['dropout']),
                                    nn.Linear(2 * self.lstm_dim, self.fc_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.fc_dim, self.num_classes))
        else:
            self.classifier = nn.Sequential(nn.Linear(2*self.lstm_dim, self.fc_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.fc_dim, self.num_classes))

    def forward(self, inp, lens):
        if not self.model_name == 'bilstm_reg':
            inp = self.embedding(inp)
            out = self.encoder(inp, lens)
            out = self.classifier(out)
        else:
            out = self.encoder(inp, self.embedding, lens)
            out = self.classifier(out.to(device))
        return out


"""
Baseline BiLSTM to run on the entire document as is:
    1. Option 1: Take the hidden layer output of final cell as the representation of the document
    2. Option 2: Take pool across embedding dimension of all the cells as the representation of the document
"""
class BiLSTM(nn.Module):
    def __init__(self, config, max_pool=False):
        super(BiLSTM, self).__init__()
        self.pool = max_pool
        self.lstm = nn.LSTM(config["embed_dim"], config["lstm_dim"], bidirectional = True)

    def forward(self, embed, length):
        sorted_len, sorted_idxs = torch.sort(length, descending =True)
        embed = embed[ : , sorted_idxs, :].to(device)

        packed_embed = pack_padded_sequence(embed, sorted_len, batch_first = False).to(device)
        all_states, hidden_states = self.lstm(packed_embed)
        all_states, _ = pad_packed_sequence(all_states, batch_first = False)

        # If not max-pool biLSTM, we extract the h0_l and h0_r from the tuple of tuples 'hn', and concat them to get the final embedding
        if not self.pool:
            out = torch.cat((hidden_states[0][0], hidden_states[0][1]), dim = 1)

        # If it is max-pooling biLSTM, set the PADS to very low numbers so that they never get selected in max-pooling
        # Then, max-pool over each dimension(which is now 2D, as 'X' = ALL) to get the final embedding
        elif self.pool:
            # replace PADs with very low numbers so that they never get picked
            out = torch.where(all_states.to('cpu') == 0, torch.tensor(-1e8), all_states.to('cpu'))
            out, _ = torch.max(out, 0)

        _, unsorted_idxs = torch.sort(sorted_idxs)
        out = out[unsorted_idxs, :].to(device)
        return out


"""
<<<<<<< HEAD
BiLSTM_regularized : BiLSTM with Temporal Averaging, weight dropout and embedding dropout. Current SOTA on Reuters
=======
BiLSTM_regularized : BiLSTM with Temporal Averaging, weight dropout and embedding dropout. Current SOTA
>>>>>>> f712068342f99b8f512a3f8bff61991c53da4ece
"""
class BiLSTM_reg(nn.Module):
    def __init__(self, config):
        super(BiLSTM_reg, self).__init__()
        self.tar = 0.0
        self.ar = 0.0
        self.beta_ema = config["beta_ema"]  # Temporal averaging
        self.wdrop = config["wdrop"]  # Weight dropping
        self.embed_droprate = config["embed_drop"]  # Embedding dropout

<<<<<<< HEAD
        self.lstm = nn.LSTM(config["embed_dim"], config["lstm_dim"], bidirectional = True, dropout=config["dropout"], num_layers=1, batch_first=False).to(device)

        # Applyying Weight dropout to hh_l0 layer of the LSTM
        weights = ['weight_hh_l0']
        self.lstm = WeightDrop(self.lstm, weights, self.wdrop).to(device)

=======
        self.lstm = nn.LSTM(config["embed_dim"], config["lstm_dim"], bidirectional = True, dropout=config["dropout"], num_layers=2, batch_first=False).to(device)
        weights = ['weight_hh_l0']
        self.lstm = WeightDrop(self.lstm, weights, self.wdrop).to(device)


        # if self.wdrop:
        #     self.lstm = WeightDrop(self.lstm, ['weight_hh_l0'], dropout=self.wdrop)
>>>>>>> f712068342f99b8f512a3f8bff61991c53da4ece
        self.dropout = nn.Dropout(config['dropout'])

        if self.beta_ema>0:
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, inp, embedding, lengths=None):
        sorted_len, sorted_idxs = torch.sort(lengths, descending =True)
        inp = inp[ : , sorted_idxs].to(device)

        inp = embedded_dropout(embedding, inp, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else embedding(inp)
        # print("Input embedding shape = ", inp.shape)

        if lengths is not None:
            inp = torch.nn.utils.rnn.pack_padded_sequence(inp, lengths, batch_first=False)
        rnn_outs, _ = self.lstm(inp)

        if lengths is not None:
            rnn_outs,_ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outs, batch_first=False)
<<<<<<< HEAD
=======
            rnn_outs_temp, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outs_temp, batch_first=False)
>>>>>>> f712068342f99b8f512a3f8bff61991c53da4ece
            # print("rnn_outs(after lstm) shape = ", rnn_outs.shape)

        out = torch.where(rnn_outs.to('cpu') == 0, torch.tensor(-1e8), rnn_outs.to('cpu'))
        out, _ = torch.max(out, 0)
        _, unsorted_idxs = torch.sort(sorted_idxs)
        out = out[unsorted_idxs, :].to(device)
<<<<<<< HEAD
=======
        if self.tar or self.ar:
            return out, rnn_outs
>>>>>>> f712068342f99b8f512a3f8bff61991c53da4ece
        return out


    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1-self.beta_ema)*p.data)
    
    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p/(1-self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p,avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params


"""
The heirarchical attention network working on word and sentence level attention
"""
class HAN(nn.Module):
    def __init__(self, config):
        super(HAN, self).__init__()


    def forward(Self, input):

        return None


"""
The (word) CNN based architecture as propsoed by Kim, et.al(2014) 
"""
class Kim_CNN(nn.Module):
    def __init__(self, config):
        super(Kim_CNN, self).__init__()


    def forward(Self, input):

        return None