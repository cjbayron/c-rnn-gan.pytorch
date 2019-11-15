# Copyright 2019 Christopher John Bayron
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been created by Christopher John Bayron based on "rnn_gan.py"
# by Olof Mogren. The referenced code is available in:
#
#     https://github.com/olofmogren/c-rnn-gan

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    ''' C-RNN-GAN generator
    '''
    def __init__(self, num_feats, hidden_units=256, drop_prob=0.6, use_cuda=False):
        super(Generator, self).__init__()

        # params
        self.hidden_dim = hidden_units
        self.use_cuda = use_cuda
        self.num_feats = num_feats

        self.fc_layer1 = nn.Linear(in_features=(num_feats*2), out_features=hidden_units)
        self.lstm_cell1 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lstm_cell2 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.fc_layer2 = nn.Linear(in_features=hidden_units, out_features=num_feats)

    def forward(self, z, states):
        ''' Forward prop
        '''
        if self.use_cuda:
            z = z.cuda()
        # z: (batch_size, seq_len, num_feats)
        # z here is the uniformly random vector
        batch_size, seq_len, num_feats = z.shape

        # split to seq_len * (batch_size * num_feats)
        z = torch.split(z, 1, dim=1)
        z = [z_step.squeeze(dim=1) for z_step in z]

        # create dummy-previous-output for first timestep
        prev_gen = torch.empty([batch_size, num_feats]).uniform_()
        if self.use_cuda:
            prev_gen = prev_gen.cuda()

        # manually process each timestep
        state1, state2 = states # (h1, c1), (h2, c2)
        gen_feats = []
        for z_step in z:
            # concatenate current input features and previous timestep output features
            concat_in = torch.cat((z_step, prev_gen), dim=-1)
            out = F.relu(self.fc_layer1(concat_in))
            h1, c1 = self.lstm_cell1(out, state1)
            h1 = self.dropout(h1) # feature dropout only (no recurrent dropout)
            h2, c2 = self.lstm_cell2(h1, state2)
            prev_gen = self.fc_layer2(h2)
            # prev_gen = F.relu(self.fc_layer2(h2)) #DEBUG
            gen_feats.append(prev_gen)

            state1 = (h1, c1)
            state2 = (h2, c2)

        # seq_len * (batch_size * num_feats) -> (batch_size * seq_len * num_feats)
        gen_feats = torch.stack(gen_feats, dim=1)

        states = (state1, state2)
        return gen_feats, states

    def init_hidden(self, batch_size):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data

        if (self.use_cuda):
            hidden = ((weight.new(batch_size, self.hidden_dim).zero_().cuda(),
                       weight.new(batch_size, self.hidden_dim).zero_().cuda()),
                      (weight.new(batch_size, self.hidden_dim).zero_().cuda(),
                       weight.new(batch_size, self.hidden_dim).zero_().cuda()))
        else:
            hidden = ((weight.new(batch_size, self.hidden_dim).zero_(),
                       weight.new(batch_size, self.hidden_dim).zero_()),
                      (weight.new(batch_size, self.hidden_dim).zero_(),
                       weight.new(batch_size, self.hidden_dim).zero_()))

        return hidden


class Discriminator(nn.Module):
    ''' C-RNN-GAN discrminator
    '''
    def __init__(self, num_feats, hidden_units=256, drop_prob=0.6, use_cuda=False):

        super(Discriminator, self).__init__()

        # params
        self.hidden_dim = hidden_units
        self.num_layers = 2
        self.use_cuda = use_cuda

        self.dropout = nn.Dropout(p=drop_prob)
        self.lstm = nn.LSTM(input_size=num_feats, hidden_size=hidden_units,
                            num_layers=self.num_layers, batch_first=True, dropout=drop_prob,
                            bidirectional=True)
        self.fc_layer = nn.Linear(in_features=(2*hidden_units), out_features=1)

    def forward(self, note_seq, state):
        ''' Forward prop
        '''
        if self.use_cuda:
            note_seq = note_seq.cuda()

        # note_seq: (batch_size, seq_len, num_feats)
        drop_in = self.dropout(note_seq) # input with dropout
        # (batch_size, seq_len, num_directions*hidden_size)
        lstm_out, state = self.lstm(drop_in, state)
        # (batch_size, seq_len, 1)
        out = self.fc_layer(lstm_out)
        out = torch.sigmoid(out)

        num_dims = len(out.shape)
        reduction_dims = tuple(range(1, num_dims))
        # (batch_size)
        out = torch.mean(out, dim=reduction_dims)

        return out, lstm_out, state

    def init_hidden(self, batch_size):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data

        layer_mult = 2 # for being bidirectional
        
        if self.use_cuda:
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
