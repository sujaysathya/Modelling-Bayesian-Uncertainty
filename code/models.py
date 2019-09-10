import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Doc_Classifier(nn.Module):
    def __init__(self, config):
        super(Doc_Classifier, self).__init__()

        self.lstm_dim = config['lstm_dim']
        self.fc_dim = config['fc_dim']
        self.num_classes = config['n_clases']

        if config['model_name'] == 'bilstm':
            self.encoder = BiLSTM(config)
        elif config['model_name'] == 'bilstm_pool':
            self.encoder = BiLSTM(config, max_pool = True)
        else:
            self.encoder = BiLSTM_attn(config)

        self.classifier = nn.Sequential(nn.Linear(2*self.lstm_dim, self.fc_dim),
                                 nn.Tanh(),
                                 nn.Linear(self.fc_dim, self.num_classes),
                                 nn.Sigmoid())

    def forward(self, inp, lens):
        out = self.encoder(inp, lens)
        out = self.classifier(out)
        return out


class BiLSTM(nn.Module):
    def __init__(self, config, max_pool=False):
        super(BiLSTM, self).__init__()
        self.pool = max_pool
        self.encoder = nn.LSTM(config["embed_dim"], config["lstm_dim"], bidirectional = True)


    def forward(self, embed, length):
        sorted_len, sorted_idxs = torch.sort(length, descending =True)
        embed = embed[ : , sorted_idxs, :]

        packed_embed = pack_padded_sequence(embed, sorted_len, batch_first = False)
        all_states, hidden_states = self.encoder(packed_embed)
        all_states, _ = pad_packed_sequence(all_states, batch_first = False)

        # If not max-pool biLSTM, we extract the h0_l and h0_r from the tuple of tuples 'hn', and concat them to get the final embedding
        if not self.pool:
            out = torch.cat((hidden_states[0][0], hidden_states[0][1]))

        # If it is max-pooling biLSTM, set the PADS to very low numbers so that they never get selected in max-pooling
        # Then, max-pool over each dimension(which is now 2D, as 'X' = ALL) to get the final embedding
        elif self.pool:
            # replace PADs with very low numbers so that they never get picked
            out = torch.where(all_states == 0, torch.tensor(-1e8), all_states)
            out, _ = torch.max(out, 0)

        _, unsorted_idxs = torch.sort(sorted_idxs)
        out = out[unsorted_idxs, :]
        return out



class BiLSTM_attn(nn.Module):
    def __init__(self, config):
        super(BiLSTM_attn, self).__init__()


    def forward(Self, input):

        return None