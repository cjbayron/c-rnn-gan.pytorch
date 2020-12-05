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

import os
from argparse import ArgumentParser
import numpy as np
import torch

from c_rnn_gan import Generator
import music_data_utils

CKPT_DIR = 'models'
G_FN = 'c_rnn_gan_g.pth'
MAX_SEQ_LEN = 256
FILENAME = 'sample.mid'

def generate(n):
    ''' Sample MIDI from trained generator model
    '''
    # prepare model
    dataloader = music_data_utils.MusicDataLoader(datadir=None)
    num_feats = dataloader.get_num_song_features()

    use_gpu = torch.cuda.is_available()
    g_model = Generator(num_feats, use_cuda=use_gpu)

    if not use_gpu:
        ckpt = torch.load(os.path.join(CKPT_DIR, G_FN), map_location='cpu')
    else:
        ckpt = torch.load(os.path.join(CKPT_DIR, G_FN))

    g_model.load_state_dict(ckpt)

    # generate from model then save to MIDI file
    g_states = g_model.init_hidden(1)
    z = torch.empty([1, MAX_SEQ_LEN, num_feats]).uniform_() # random vector
    if use_gpu:
        z = z.cuda()
        g_model.cuda()

    g_model.eval()

    full_song_data = []
    for i in range(n):
        g_feats, g_states = g_model(z, g_states)
        song_data = g_feats.squeeze().cpu()
        song_data = song_data.detach().numpy() 
        full_song_data.append(song_data)

    if len(full_song_data) > 1:
        full_song_data = np.concatenate(full_song_data, axis=0)
    else:
        full_song_data = full_song_data[0]

    dataloader.save_data(FILENAME, full_song_data)
    print('Full sequence shape: ', full_song_data.shape)
    print('Generated {}'.format(FILENAME))


if __name__ == "__main__":
    ARG_PARSER = ArgumentParser()
    # number of times to execute generator model;
    # all generated data are concatenated to form a single longer sequence
    ARG_PARSER.add_argument('-n', default=1, type=int)
    ARGS = ARG_PARSER.parse_args()

    generate(ARGS.n)
