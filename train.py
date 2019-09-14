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

import torch
import torch.nn as nn
from torch import optim

from c_rnn_gan import Generator, Discriminator
import music_data_utils

DATA_DIR = 'data'
CKPT_DIR = 'models'
G_FN = 'c_rnn_gan_g.pth'
D_FN = 'c_rnn_gan_d.pth'

G_LRN_RATE = 0.001
D_LRN_RATE = 0.001
MAX_GRAD_NORM = 5.0
BATCH_SIZE = 32
MAX_EPOCHS = 500
L2_DECAY = 1.0

COMPOSER = 'mozart'
MAX_SEQ_LEN = 200

PERFORM_LOSS_CHECKING = False
FREEZE_G = False
FREEZE_D = False


class DLoss(nn.Module):
    ''' C-RNN-GAN discriminator loss
    '''
    def __init__(self):
        super(DLoss, self).__init__()

    def forward(self, logits_real, logits_gen):
        ''' Discriminator loss

        logits_real: logits from D, when input is real
        logits_gen: logits from D, when input is from Generator
        '''
        logits_real = torch.clamp(logits_real, 1e-1000000, 1.0)
        logits_gen = torch.clamp(logits_gen, 0.0, 1.0-1e-1000000)

        d_loss_real = -torch.log(logits_real)
        d_loss_gen = -torch.log(1 - logits_gen)

        batch_loss = d_loss_real + d_loss_gen
        return torch.mean(batch_loss)


def control_grad(model, freeze=True):
    ''' Freeze/unfreeze optimization of model
    '''
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    else: # unfreeze
        for param in model.parameters():
            param.requires_grad = True 


def check_loss(model, loss):
    ''' Check loss and control gradients if necessary
    '''
    control_grad(model['g'], freeze=False)
    control_grad(model['d'], freeze=False)  

    if loss['d'] == 0.0 and loss['g'] == 0.0:
        print('Both G and D train loss are zero. Exiting.')
        return False
    elif loss['d'] == 0.0: # freeze D
        control_grad(model['d'], freeze=True)
    elif loss['g'] == 0.0: # freeze G
        control_grad(model['g'], freeze=True)
    elif loss['g'] < 2.0 or loss['d'] < 2.0:
        control_grad(model['d'], freeze=True)
        if loss['g']*0.7 > loss['d']:
            control_grad(model['g'], freeze=True)

    return True


def run_training(model, optimizer, criterion, dataloader):
    ''' Run single epoch
    '''
    loss = {
        'g': 10.0,
        'd': 10.0
    }

    num_feats = dataloader.get_num_song_features()
    dataloader.rewind(part='train')
    batch_meta, batch_song = dataloader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part='train')

    model['g'].train()
    model['d'].train()

    g_loss_total = 0.0
    d_loss_total = 0.0
    num_sample = 0

    while batch_meta is not None and batch_song is not None:

        real_batch_sz = batch_song.shape[0]

        # loss checking
        if PERFORM_LOSS_CHECKING == True:
            if not check_loss(model, loss):
                break

        # get initial states
        # each batch is independent i.e. not a continuation of previous batch
        # so we reset states for each batch
        # POSSIBLE IMPROVEMENT: next batch is continuation of previous batch
        g_states = model['g'].init_hidden(real_batch_sz)
        d_state = model['d'].init_hidden(real_batch_sz)

        #### GENERATOR ####
        if not FREEZE_G:
            optimizer['g'].zero_grad()
        # prepare inputs
        z = torch.empty([real_batch_sz, MAX_SEQ_LEN, num_feats]).uniform_() # random vector
        batch_song = torch.Tensor(batch_song)

        # feed inputs to generator
        g_feats, _ = model['g'](z, g_states)
        # feed real and generated input to discriminator
        _, d_feats_real, _ = model['d'](batch_song, d_state)
        _, d_feats_gen, _ = model['d'](g_feats, d_state)
        # calculate loss, backprop, and update weights of G
        loss['g'] = criterion['g'](d_feats_real, d_feats_gen)
        if not FREEZE_G:
            loss['g'].backward()
            nn.utils.clip_grad_norm_(model['g'].parameters(), max_norm=MAX_GRAD_NORM)
            optimizer['g'].step()

        #### DISCRIMINATOR ####
        if not FREEZE_D:
            optimizer['d'].zero_grad()
        # feed real and generated input to discriminator
        d_logits_real, _, _ = model['d'](batch_song, d_state)
        # need to detach from operation history to prevent backpropagating to generator
        d_logits_gen, _, _ = model['d'](g_feats.detach(), d_state)
        # calculate loss, backprop, and update weights of D
        loss['d'] = criterion['d'](d_logits_real, d_logits_gen)
        if not FREEZE_D:
            loss['d'].backward()
            nn.utils.clip_grad_norm_(model['d'].parameters(), max_norm=MAX_GRAD_NORM)
            optimizer['d'].step()

        g_loss_total += loss['g'].item()
        d_loss_total += loss['d'].item()
        num_sample += real_batch_sz

        # fetch next batch
        batch_meta, batch_song = dataloader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part='train')

    g_loss_avg, d_loss_avg = 0.0, 0.0
    if num_sample > 0:
        g_loss_avg = g_loss_total / num_sample
        d_loss_avg = d_loss_total / num_sample

    return model, g_loss_avg, d_loss_avg


def run_validation(model, criterion, dataloader):
    '''
    '''
    num_feats = dataloader.get_num_song_features()
    dataloader.rewind(part='validation')
    batch_meta, batch_song = dataloader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part='validation')

    model['g'].eval()
    model['d'].eval()

    g_loss_total = 0.0
    d_loss_total = 0.0
    num_sample = 0

    while batch_meta is not None and batch_song is not None:

        real_batch_sz = batch_song.shape[0]

        # initial states
        g_states = model['g'].init_hidden(real_batch_sz)
        d_state = model['d'].init_hidden(real_batch_sz)

        #### GENERATOR ####
        # prepare inputs
        z = torch.empty([real_batch_sz, MAX_SEQ_LEN, num_feats]).uniform_() # random vector
        batch_song = torch.Tensor(batch_song)

        # feed inputs to generator
        g_feats, _ = model['g'](z, g_states)
        # feed real and generated input to discriminator
        d_logits_real, d_feats_real, _ = model['d'](batch_song, d_state)
        d_logits_gen, d_feats_gen, _ = model['d'](g_feats, d_state)
        # calculate loss
        g_loss = criterion['g'](d_feats_real, d_feats_gen)
        d_loss = criterion['d'](d_logits_real, d_logits_gen)

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()
        num_sample += real_batch_sz

        # fetch next batch
        batch_meta, batch_song = dataloader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part='validation')

    g_loss_avg, d_loss_avg = 0.0, 0.0
    if num_sample > 0:
        g_loss_avg = g_loss_total / num_sample
        d_loss_avg = d_loss_total / num_sample

    return g_loss_avg, d_loss_avg


def main(args):
    ''' Training sequence
    '''
    dataloader = music_data_utils.MusicDataLoader(DATA_DIR, single_composer=COMPOSER)
    num_feats = dataloader.get_num_song_features()

    # First checking if GPU is available
    train_on_gpu = torch.cuda.is_available()
    if(train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    model = {
        'g': Generator(num_feats, use_cuda=train_on_gpu),
        'd': Discriminator(num_feats, use_cuda=train_on_gpu)
    }

    optimizer = {
        # 'g': optim.SGD(model['g'].parameters(), G_LRN_RATE, weight_decay=L2_DECAY),
        'g': optim.Adam(model['g'].parameters(), G_LRN_RATE),
        'd': optim.Adam(model['d'].parameters(), D_LRN_RATE)
    }

    criterion = {
        'g': nn.MSELoss(reduction='sum'), # feature matching
        'd': DLoss()
    }

    if args.load_g:
        ckpt = torch.load(os.path.join(CKPT_DIR, G_FN))
        model['g'].load_state_dict(ckpt)
        print("Continue training of %s" % os.path.join(CKPT_DIR, G_FN))

    if args.load_d:
        ckpt = torch.load(os.path.join(CKPT_DIR, D_FN))
        model['d'].load_state_dict(ckpt)
        print("Continue training of %s" % os.path.join(CKPT_DIR, D_FN))

    if(train_on_gpu):
        model['g'].cuda()
        model['d'].cuda()

    for ep in range(args.num_epochs):
        model, trn_g_loss, trn_d_loss = run_training(model, optimizer, criterion, dataloader)
        val_g_loss, val_d_loss = run_validation(model, criterion, dataloader)

        print("Epoch %d/%d [Training Loss] G: %0.8f, D: %0.8f "
              "[Validation Loss] G: %0.8f, D: %0.8f" %
              (ep+1, args.num_epochs, trn_g_loss, trn_d_loss, val_g_loss, val_d_loss))

        # sampling (to check if generator really learns)

    if not args.no_save_g:
        torch.save(model['g'].state_dict(), os.path.join(CKPT_DIR, G_FN))
        print("Saved generator: %s" % os.path.join(CKPT_DIR, G_FN))

    if not args.no_save_d:
        torch.save(model['d'].state_dict(), os.path.join(CKPT_DIR, D_FN))
        print("Saved discriminator: %s" % os.path.join(CKPT_DIR, D_FN))


if __name__ == "__main__":

    ARG_PARSER = ArgumentParser()
    ARG_PARSER.add_argument('--load_g', action='store_true')
    ARG_PARSER.add_argument('--load_d', action='store_true')
    ARG_PARSER.add_argument('--no_save_g', action='store_true')
    ARG_PARSER.add_argument('--no_save_d', action='store_true')
    ARG_PARSER.add_argument('--num_epochs', default=500, type=int)
    ARG_PARSER.add_argument('--seq_len', default=256, type=int)
    ARG_PARSER.add_argument('--batch_size', default=32, type=int)

    ARGS = ARG_PARSER.parse_args()
    MAX_SEQ_LEN = ARGS.seq_len
    BATCH_SIZE = ARGS.batch_size

    main(ARGS)