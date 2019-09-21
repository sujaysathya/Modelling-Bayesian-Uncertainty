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
import torch.nn.functional as F

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





class WeightDrop_manual(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def null_function(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.null_function

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

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

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)




###################
## Model Classes ##
###################

"""
Main class that controls training and calling of other classes based on corresponding model_name

"""
class Document_Classifier(nn.Module):
    def __init__(self, config, pre_trained_embeds = None):
        super(Document_Classifier, self).__init__()

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
            self.fc_inp_dim = config['lstm_dim']
        elif self.model_name == 'bilstm_pool':
            self.encoder = BiLSTM(config, max_pool = True)
            self.fc_inp_dim = config['lstm_dim']
        elif self.model_name == 'bilstm_reg':
            self.encoder = BiLSTM_reg(config)
            self.fc_inp_dim = config['lstm_dim']
        elif self.model_name == 'han':
            self.encoder = HAN(config)
            self.fc_inp_dim = config['sent_gru_dim']
        elif self.model_name == 'cnn':
            self.encoder = Kim_CNN(config)
            self.fc_inp_dim = int(config["kernel_num"]*len(config["kernel_sizes"].split(','))/2)


        if self.model_name in ['bilstm' , 'bilstm_pool']:
            self.classifier = nn.Sequential(nn.Linear(2*self.fc_inp_dim, self.fc_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.fc_dim, self.num_classes))

    def forward(self, inp, lens):
        if self.model_name in ['bilstm' , 'bilstm_pool']:
            inp = self.embedding(inp)
            out = self.encoder(inp, lens)
            out = self.classifier(out)
        else:
            out = self.encoder(inp, self.embedding, lens)
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
BiLSTM_regularized : BiLSTM with Temporal Averaging, weight dropout and embedding dropout. Current SOTA on Reuters dataset
"""
class BiLSTM_reg(nn.Module):
    def __init__(self, config):
        super(BiLSTM_reg, self).__init__()
        self.tar = 0.0
        self.ar = 0.0
        self.beta_ema = config["beta_ema"]  # Temporal averaging
        self.wdrop = config["wdrop"]  # Weight dropping
        self.embed_droprate = config["embed_drop"]  # Embedding dropouts
        self.dropout = config['dropout']
        self.lstm_dim = config['lstm_dim']
        self.embed_dim = config['embed_dim']
        self.num_classes = config['n_classes']

        self.lstm = nn.LSTM(self.embed_dim, self.lstm_dim, bidirectional = True, dropout=config["dropout"], num_layers=1, batch_first=False).to(device)
        self.classifier = nn.Sequential(nn.Dropout(self.dropout) , nn.Linear(2*self.lstm_dim, 2*self.lstm_dim), nn.ReLU(), nn.Linear(2*self.lstm_dim, self.num_classes))

        # Applyying Weight dropout to hh_l0 layer of the LSTM
        weights = ['weight_hh_l0']
        self.lstm = WeightDrop(self.lstm, weights, self.wdrop).to(device)

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
        all_states, _ = self.lstm(inp)

        if lengths is not None:
            all_states,_ = torch.nn.utils.rnn.pad_packed_sequence(all_states, batch_first=False)
            # print("rnn_outs(after lstm) shape = ", rnn_outs.shape)

        out = torch.where(all_states.to('cpu') == 0, torch.tensor(-1e8), all_states.to('cpu'))
        out, _ = torch.max(out, 0)
        _, unsorted_idxs = torch.sort(sorted_idxs)
        out = out[unsorted_idxs, :].to(device)
        out = self.classifier(out)
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



####################################
## Classes for HAN implementation ##
####################################
"""
The heirarchical attention network working on word and sentence level attention
"""
# class HAN_main(nn.Module):
#     def __init__(self, config):
#         super(HAN_main, self).__init__()
#         self.word_attn = HAN_word_attention(config)
#         self.sentence_attn = HAN_sentence_attention(config)
#         self.classifier = nn.Linear(2*config['sent_gru_dim'], config['n_classes'])

#     def forward(self, inp, embedding, length):
#         inp = inp.permute(1,2,0)
#         num_sents = inp.size(0)
#         sent_representations = None
#         for i in range(num_sents):
#             word_attn_outs = self.word_attn(inp[i, :], embedding)
#             if sent_representations is None:
#                 sent_representations = word_attn_outs
#             else:
#                 torch.cat((sent_representations, word_attn_outs), dim=0)

#         sent_attn_outs = self.sentence_attn(sent_representations)
#         out = self.classifier(sent_attn_outs)
#         return out


# class HAN_word_attention(nn.Module):
#     def __init__(self, config):
#         super(HAN_word_attention, self).__init__()
#         self.word_hidden_dim = config['word_gru_dim']
#         self.embed_dim = config['embed_dim']
#         self.context_weights = nn.Parameter(torch.rand(2*self.word_hidden_dim, 1))
#         self.context_weights.data.uniform_(-0.25, 0.25)
#         self.gru = nn.GRU(self.embed_dim, self.word_hidden_dim, bidirectional = True)
#         self.lin_projection = nn.Sequential(nn.Linear(2*self.word_hidden_dim, 2*self.word_hidden_dim), nn.Tanh())
#         self.attn_wts = nn.Softmax()

#     def forward(self, inp, embedding):
#         # print(inp.shape)
#         inp = embedding(inp)
#         # print(inp.shape)
#         all_states_words, _ = self.gru(inp)
#         # print(all_states_words.shape)
#         out = self.lin_projection(all_states_words)
#         # print(out.shape)
#         out = torch.matmul(out, self.context_weights)
#         # print(out.shape)
#         out = out.squeeze(dim=2)
#         # print(out.shape)
#         out = self.attn_wts(out.transpose(1, 0))
#         # print(out.shape)
#         out = torch.mul(all_states_words.permute(2, 0, 1), out.transpose(1, 0))
#         # print(out.shape)
#         out = torch.sum(out, dim=1).transpose(1, 0).unsqueeze(0)
#         # print(out.shape)
#         return out


# class HAN_sentence_attention(nn.Module):
#     def __init__(self, config):
#         super(HAN_sentence_attention, self).__init__()
#         self.word_hidden_dim = config['word_gru_dim']
#         self.sent_hidden_dim = config['sent_gru_dim']
#         self.embed_dim = config['embed_dim']
#         self.context_weights = nn.Parameter(torch.rand(2*self.sent_hidden_dim, 1))
#         self.context_weights.data.uniform_(-0.1, 0.1)
#         self.gru = nn.GRU(2*self.word_hidden_dim, self.sent_hidden_dim, bidirectional = True)
#         self.lin_projection = nn.Sequential(nn.Linear(2*self.sent_hidden_dim, 2*self.sent_hidden_dim), nn.Tanh())
#         self.attn_wts = nn.Softmax()

#     def forward(self, inp):
#         all_states_sents,_ = self.gru(inp)
#         out = self.lin_projection((all_states_sents))
#         out = torch.matmul(out, self.context_weights)
#         out = out.squeeze(dim=2)
#         out = self.attn_wts(out.transpose(1,0))
#         out = torch.mul(all_states_sents.permute(2, 0, 1), out.transpose(1, 0))
#         out = torch.sum(out, dim=1).transpose(1, 0).unsqueeze(0)
#         return out.squeeze(0)


class HAN(nn.Module):
    def __init__(self, config):
        super(HAN, self).__init__()
        self.word_hidden_dim = config['word_gru_dim']
        self.embed_dim = config['embed_dim']
        self.num_classes = config['n_classes']

        # Word attention
        self.word_context_weights = nn.Parameter(torch.rand(2*self.word_hidden_dim, 1))
        self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.word_attn_gru = nn.GRU(self.embed_dim, self.word_hidden_dim, bidirectional = True)
        self.word_lin_projection = nn.Sequential(nn.Linear(2*self.word_hidden_dim, 2*self.word_hidden_dim), nn.Tanh())
        self.word_attn_wts = nn.Softmax()

        # Sentence attention
        self.sent_hidden_dim = config['sent_gru_dim']
        self.sent_context_weights = nn.Parameter(torch.rand(2*self.sent_hidden_dim, 1))
        self.sent_context_weights.data.uniform_(-0.1, 0.1)
        self.sentence_attn_gru = nn.GRU(2*self.word_hidden_dim, self.sent_hidden_dim, bidirectional = True, batch_first = False)
        self.sent_lin_projection = nn.Sequential(nn.Linear(2*self.sent_hidden_dim, 2*self.sent_hidden_dim), nn.Tanh())
        self.sent_attn_wts = nn.Softmax()

        self.classifier = nn.Linear(2*self.sent_hidden_dim, self.num_classes)

    def forward(self, inp, embedding, length):
        inp = inp.permute(1,2,0)
        num_sents = inp.size(0)
        sent_representations = None

        # Word-attention block
        for i in range(num_sents):
            model_inp = inp[i, :]
            model_inp = embedding(model_inp)
            all_states_words, _ = self.word_attn_gru(model_inp)
            out = self.word_lin_projection(all_states_words)
            out = torch.matmul(out, self.word_context_weights)
            out = out.squeeze(dim=2)
            out = self.word_attn_wts(out.transpose(1, 0))
            out = torch.mul(all_states_words.permute(2, 0, 1), out.transpose(1, 0))
            word_attn_outs = torch.sum(out, dim=1).transpose(1, 0).unsqueeze(0)
            if sent_representations is None:
                sent_representations = word_attn_outs
            else:
                sent_representations =  torch.cat((sent_representations, word_attn_outs), dim=0)

        # Sentence-attention Block
        all_states_sents,_ = self.sentence_attn_gru(sent_representations)
        out = self.sent_lin_projection((all_states_sents))
        out = torch.matmul(out, self.sent_context_weights)
        out = out.squeeze(dim=2)
        out = self.sent_attn_wts(out.transpose(1,0))
        out = torch.mul(all_states_sents.permute(2, 0, 1), out.transpose(1, 0))
        out = torch.sum(out, dim=1).transpose(1, 0).unsqueeze(0)
        out = out.squeeze(0)
        sent_attn_out = self.classifier(out)
        return sent_attn_out


"""
The (word) CNN based architecture as propsoed by Kim, et.al(2014) 
"""
class Kim_CNN(nn.Module):
    def __init__(self, config):
        super(Kim_CNN, self).__init__()
        self.embed_dim = config["embed_dim"]
        self.num_classes = config["n_classes"]
        self.input_channels = 1
        self.num_kernels = config["kernel_num"]
        self.kernel_sizes = [int(k) for k in config["kernel_sizes"].split(',')]
        self.fc_inp_dim = self.num_kernels * len(self.kernel_sizes)
        self.fc_dim = config['fc_dim']

        self.cnn = nn.ModuleList([nn.Conv2d(self.input_channels, self.num_kernels, (k_size, self.embed_dim)) for k_size in self.kernel_sizes])
        self.classifier = nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(self.fc_inp_dim, self.fc_dim), nn.ReLU(), nn.Linear(self.fc_dim, self.num_classes))


    def forward(self, inp, embedding, lengths=None):
        # x is (B, L, D)
        inp = embedding(inp)
        inp = inp.permute(1,0,2)
        inp = inp.unsqueeze(1)  # (B, Ci, L, D)
        inp = [F.relu(conv(inp)).squeeze(3) for conv in self.cnn]  # [(B, Co, L), ...]*len(Ks)
        inp = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inp]  # [(B, Co), ...]*len(Ks)
        out = torch.cat(inp, 1) # (B, len(Ks)*Co)
        out = self.classifier(out)
        return out