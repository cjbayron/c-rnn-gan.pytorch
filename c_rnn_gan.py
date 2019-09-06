import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, z_size, vec_size, hidden_units=128, lstm_dim=350, drop_prob=0.6):
        super(Generator, self).__init__()

        # params
        self.num_layers = 2
        self.vec_size = vec_size

        self.fc_layer1 = nn.Linear(in_features=z_size, out_features=hidden_units)
        self.lstm = nn.LSTM(input_size=hidden_units, hidden_size=lstm_dim,
                            num_layers=self.num_layers, batch_first=True, dropout=drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.fc_layer2 = nn.Linear(in_features=lstm_dim, out_features=vec_size)

    def forward(z, hidden):
        batch_size = z.shape[0]
        # z: (batch_size, seq_len, z_size)
        out = self.fc_layer1(z)
        out = F.leaky_relu(out)
        out = self.lstm(out, hidden)
        # stack outputs before feeding to fully-connected
        out = out.contiguous().view(-1, self.lstm_dim)
        out = self.dropout(out)
        out = self.fc_layer(out)
        # (batch_size * seq_len, vec_size)
        notes = out.view(batch_size, -1, self.vec_size)

        return notes, hidden

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

    def __init__(self, vec_size, seq_len, lstm_dim=350, drop_prob=0.6):

        super(Discriminator, self).__init__()

        # params
        self.num_layers = 2
        self.bidirectional = True
        self.seq_len = seq_len
        self.lstm_dim = lstm_dim

        self.lstm = nn.LSTM(input_size=vec_size, hidden_size=lstm_dim,
                            num_layers=self.num_layers, batch_first=True, dropout=drop_prob,
                            bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(drop_prob)
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
        drop_out = self.dropout(lstm_out)
        fc_out = self.fc_layer(drop_out)
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
