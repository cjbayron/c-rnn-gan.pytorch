import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, num_feats, hidden_units=128, lstm_dim=256, drop_prob=0.6):
        super(Generator, self).__init__()

        # params
        self.fc_layer1 = nn.Linear(in_features=(z_size*2), out_features=hidden_units)
        self.lstm_cell1 = nn.LSTMCell(input_size=hidden_units, hidden_size=lstm_dim)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lstm_cell2 = nn.LSTMCell(input_size=lstm_dim, hidden_size=lstm_dim)
        self.fc_layer2 = nn.Linear(in_features=lstm_dim, out_features=num_feats)

    def forward(z, states):
        # z: (batch_size, seq_len, z_size)
        # z here is the uniformly random vector
        batch_size, seq_len, z_size = z.shape

        # split to seq_len * (batch_size * z_size)
        z = torch.split(z, 1, dim=1)
        z = [z_step.squeeze() for z_step in z]

        # create dummy-previous-output for first timestep
        prev_gen = torch.empty([batch_size, z_size]).uniform_()

        # manually process each timestep
        state1, state2 = states
        gen_feats = []
        for z_step in z:
            # concatenate current input features and previous timestep output features
            concat_in = torch.cat((z_step, prev_gen), dim=-1)
            out = F.relu(self.fc_layer1(concat_in))
            out, state1 = self.lstm_cell1(out, state1)
            out = self.dropout(out) # feature dropout only (no recurrent dropout)
            out, state2 = self.lstm_cell2(out, state2)
            prev_gen = self.fc_layer2(out)
            gen_feats.append(prev_gen)

        states = (state1, state2)
        return gen_feats, states

    def init_hidden(self, batch_size, train_on_gpu=False):
        ''' Initialize hidden state '''
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.num_layers, batch_size,
                                 self.hidden_dim).zero_().cuda(),
                      weight.new(self.num_layers, batch_size,
                                 self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers, batch_size,
                                 self.hidden_dim).zero_(),
                      weight.new(self.num_layers, batch_size,
                                 self.hidden_dim).zero_())
        
        return hidden


class Discriminator(nn.Module):

    def __init__(self, vec_size, seq_len, lstm_dim=256, lstm_layers=2, drop_prob=0.6):

        super(Discriminator, self).__init__()

        # params
        self.num_layers = lstm_layers
        self.bidirectional = True
        self.seq_len = seq_len
        self.lstm_dim = lstm_dim

        # NOTE: original implementation uses dropout on input
        self.lstm = nn.LSTM(input_size=vec_size, hidden_size=lstm_dim,
                            num_layers=self.num_layers, batch_first=True, dropout=drop_prob,
                            bidirectional=self.bidirectional)
        self.fc_layer = nn.Linear(in_features=lstm_dim, out_features=1)

    def forward(note_seq, hidden):
        # note_seq: (batch_size, seq_len, vec_size)
        lstm_out = self.lstm(note_seq, hidden)

        if self.bidirectional:
            # separate to forward and backward
            lstm_out = lstm_out.contiguous().view(-1, self.seq_len, 2, self.lstm_dim)
            # get backward output in first node
            lstm_out_bw = lstm_out[:, 0, 1, :]
            # get forward output in last node
            lstm_out_fw = lstm_out[:, -1, 0, :]
            # average outputs
            lstm_out = torch.add(input=lstm_out_bw, alpha=1, other=lstm_out_fw)
            lstm_out = torch.div(lstm_out, 2)
        else:
            # if unidirectional, get only last cell output
            lstm_out = lstm_out[:, -1]

        # stack outputs before feeding to fully-connected
        lstm_out = lstm_out.contiguous().view(-1, self.lstm_dim)
        fc_out = self.fc_layer(lstm_out)
        sig_out = torch.sigmoid(fc_out)
        # sig_out: (batch_size, 1)

        return sig_out, hidden

    def init_hidden(self, batch_size, train_on_gpu=False):
        ''' Initialize hidden state '''
        weight = next(self.parameters()).data
        layer_mult = 1
        if self.bidirectional:
            layer_mult *= 2 
        
        if (train_on_gpu):
            hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_().cuda(),
                      weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_(),
                      weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_())
        
        return hidden
