import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, num_feats, hidden_units=128, lstm_dim=256, drop_prob=0.6):
        super(Generator, self).__init__()

        # params
        self.fc_layer1 = nn.Linear(in_features=(num_feats*2), out_features=hidden_units)
        self.lstm_cell1 = nn.LSTMCell(input_size=hidden_units, hidden_size=lstm_dim)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lstm_cell2 = nn.LSTMCell(input_size=lstm_dim, hidden_size=lstm_dim)
        self.fc_layer2 = nn.Linear(in_features=lstm_dim, out_features=num_feats)

    def forward(z, states):
        # z: (batch_size, seq_len, num_feats)
        # z here is the uniformly random vector
        batch_size, seq_len, num_feats = z.shape

        # split to seq_len * (batch_size * num_feats)
        z = torch.split(z, 1, dim=1)
        z = [z_step.squeeze() for z_step in z]

        # create dummy-previous-output for first timestep
        prev_gen = torch.empty([batch_size, num_feats]).uniform_()

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

        # seq_len * (batch_size * num_feats) -> (batch_size * seq_len * num_feats)
        gen_feats = torch.stack(gen_feats, dim=1)

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

    def __init__(self, num_feats, lstm_dim=256, drop_prob=0.6):

        super(Discriminator, self).__init__()

        # params
        self.dropout = nn.Dropout(p=drop_prob)
        self.lstm = nn.LSTM(input_size=num_feats, hidden_size=lstm_dim,
                            num_layers=2, batch_first=True, dropout=drop_prob,
                            bidirectional=True)
        self.fc_layer = nn.Linear(in_features=(2*lstm_dim), out_features=1)

    def forward(note_seq, state):
        # note_seq: (batch_size, seq_len, num_feats)
        drop_in = self.dropout(note_seq) # input with dropout
        # (batch_size, seq_len, num_directions*hidden_size)
        lstm_out = self.lstm(drop_in, state)
        # (batch_size, seq_len, 1)
        out = self.fc_layer(lstm_out)
        out = torch.sigmoid(out)

        num_dims = len(out.shape)
        reduction_dims = tuple(range(1, num_dims))
        # (batch_size)
        out = torch.mean(out, dim=reduction_dims)

        return out, lstm_out, state

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
