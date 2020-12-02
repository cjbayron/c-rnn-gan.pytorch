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
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

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

NUM_DUMMY_TRN = 256
NUM_DUMMY_VAL = 128

EPSILON = 1e-40 # value to use to approximate zero (to prevent undefined results)

def get_accuracy(logits_real, logits_gen):
    ''' Discriminator accuracy
    '''
    real_corrects = (logits_real > 0.5).sum()
    gen_corrects = (logits_gen < 0.5).sum()

    acc = (real_corrects + gen_corrects) / (len(logits_real) + len(logits_gen))
    return acc.item()

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
        logits_real = torch.clamp(logits_real, EPSILON, 1.0)
        d_loss_real = -torch.log(logits_real)

        logits_gen = torch.clamp((1 - logits_gen), EPSILON, 1.0)
        d_loss_gen = -torch.log(logits_gen)

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


def dummy_dataloader(seq_len, batch_size, num_sample):
    ''' Dummy data generator (for debugging purposes)
    '''
    # the following code generates random data of numbers
    # where each number is twice the prev number
    np_data = np.stack([(2 ** np.arange(seq_len))[:, np.newaxis] \
                        * np.random.rand() for i in range(num_sample)])

    data = TensorDataset(torch.from_numpy(np_data))
    return DataLoader(data, shuffle=True, batch_size=batch_size)


def run_training(model, optimizer, criterion, dataloader, ep, freeze_g=False, freeze_d=False):
    ''' Run single training epoch
    '''

    loss = {
        'g': 10.0,
        'd': 10.0
    }

    num_feats = model['g'].num_feats

    model['g'].train()
    model['d'].train()

    g_loss_total = 0.0
    d_loss_total = 0.0
    num_corrects = 0
    num_sample = 0

    log_sum_real = 0.0
    log_sum_gen = 0.0

    for (batch_input, ) in dataloader:

        real_batch_sz = len(batch_input)
        batch_input = batch_input.type(torch.FloatTensor)

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
        if not freeze_g:
            optimizer['g'].zero_grad()
        # prepare inputs
        z = torch.empty([real_batch_sz, MAX_SEQ_LEN, num_feats]).uniform_() # random vector

        # feed inputs to generator
        g_feats, _ = model['g'](z, g_states)
        # feed real and generated input to discriminator
        _, d_feats_real, _ = model['d'](batch_input, d_state)
        _, d_feats_gen, _ = model['d'](g_feats, d_state)

        # calculate loss, backprop, and update weights of G
        loss['g'] = criterion['g'](d_feats_real, d_feats_gen)
        if not freeze_g:
            loss['g'].backward()
            # nn.utils.clip_grad_norm_(model['g'].parameters(), max_norm=MAX_GRAD_NORM)
            optimizer['g'].step()

        #### DISCRIMINATOR ####
        if not freeze_d:
            optimizer['d'].zero_grad()

        # feed real and generated input to discriminator
        d_logits_real, _, _ = model['d'](batch_input, d_state)
        # need to detach from operation history to prevent backpropagating to generator
        d_logits_gen, _, _ = model['d'](g_feats.detach(), d_state)
        # calculate loss, backprop, and update weights of D
        loss['d'] = criterion['d'](d_logits_real, d_logits_gen)

        # print("Trn: ", d_logits_real.mean(), d_logits_gen.mean())
        log_sum_real += d_logits_real.sum().item()
        log_sum_gen += d_logits_gen.sum().item()

        if not freeze_d:
            loss['d'].backward()
            nn.utils.clip_grad_norm_(model['d'].parameters(), max_norm=MAX_GRAD_NORM)
            optimizer['d'].step()

        g_loss_total += loss['g'].item()
        d_loss_total += loss['d'].item()
        num_corrects += (d_logits_real > 0.5).sum().item() + (d_logits_gen < 0.5).sum().item()
        num_sample += real_batch_sz

    g_loss_avg, d_loss_avg = 0.0, 0.0
    d_acc = 0.0
    if num_sample > 0:
        g_loss_avg = g_loss_total / num_sample
        d_loss_avg = d_loss_total / num_sample
        d_acc = 100 * num_corrects / (2 * num_sample) # 2 because (real + generated)

        print("Trn: ", log_sum_real / num_sample, log_sum_gen / num_sample)

    return model, g_loss_avg, d_loss_avg, d_acc


def run_validation(model, criterion, dataloader):
    ''' Run single validation epoch
    '''
    num_feats = model['g'].num_feats

    model['g'].eval()
    model['d'].eval()

    g_loss_total = 0.0
    d_loss_total = 0.0
    num_corrects = 0
    num_sample = 0

    log_sum_real = 0.0
    log_sum_gen = 0.0

    for (batch_input, ) in dataloader:

        real_batch_sz = len(batch_input)
        batch_input = batch_input.type(torch.FloatTensor)

        # initial states
        g_states = model['g'].init_hidden(real_batch_sz)
        d_state = model['d'].init_hidden(real_batch_sz)

        #### GENERATOR ####
        # prepare inputs
        z = torch.empty([real_batch_sz, MAX_SEQ_LEN, num_feats]).uniform_() # random vector

        # feed inputs to generator
        g_feats, _ = model['g'](z, g_states)
        # feed real and generated input to discriminator
        d_logits_real, d_feats_real, _ = model['d'](batch_input, d_state)
        d_logits_gen, d_feats_gen, _ = model['d'](g_feats, d_state)
        # print("Val: ", d_logits_real.mean(), d_logits_gen.mean())
        log_sum_real += d_logits_real.sum().item()
        log_sum_gen += d_logits_gen.sum().item()

        # calculate loss
        g_loss = criterion['g'](d_feats_real, d_feats_gen)
        d_loss = criterion['d'](d_logits_real, d_logits_gen)

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()
        num_corrects += (d_logits_real > 0.5).sum().item() + (d_logits_gen < 0.5).sum().item()
        num_sample += real_batch_sz

    g_loss_avg, d_loss_avg = 0.0, 0.0
    d_acc = 0.0
    if num_sample > 0:
        g_loss_avg = g_loss_total / num_sample
        d_loss_avg = d_loss_total / num_sample
        d_acc = 100 * num_corrects / (2 * num_sample) # 2 because (real + generated)

        print("Val: ", log_sum_real / num_sample, log_sum_gen / num_sample)

    return g_loss_avg, d_loss_avg, d_acc


def generate_sample(g_model, num_sample=1):
    ''' Sample from generator
    '''
    num_feats = g_model.num_feats
    g_states = g_model.init_hidden(num_sample)

    z = torch.empty([num_sample, MAX_SEQ_LEN, num_feats]).uniform_() # random vector

    g_feats, _ = g_model(z, g_states)
    return g_feats


def main(args):
    ''' Training sequence
    '''
    trn_dataloader = dummy_dataloader(MAX_SEQ_LEN, BATCH_SIZE, NUM_DUMMY_TRN)
    val_dataloader = dummy_dataloader(MAX_SEQ_LEN, BATCH_SIZE, NUM_DUMMY_VAL)

    # First checking if GPU is available
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    model = {
        'g': Generator(num_feats=1, use_cuda=train_on_gpu),
        'd': Discriminator(num_feats=1, use_cuda=train_on_gpu)
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

    if train_on_gpu:
        model['g'].cuda()
        model['d'].cuda()

    if not args.no_pretraining:
        for ep in range(args.pretraining_epochs):
            model, trn_g_loss, trn_d_loss, trn_acc = \
                run_training(model, optimizer, criterion, trn_dataloader, ep, freeze_g=True)
            val_g_loss, val_d_loss, val_acc = run_validation(model, criterion, val_dataloader)

            sample = generate_sample(model['g'])

            print("Epoch %d/%d\n"
                  "\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
                  "\t[Validation] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f" %
                  (ep+1, args.num_epochs, trn_g_loss, trn_d_loss, trn_acc,
                   val_g_loss, val_d_loss, val_acc))

            print(sample)

        for ep in range(args.pretraining_epochs):
            model, trn_g_loss, trn_d_loss, trn_acc = \
                run_training(model, optimizer, criterion, trn_dataloader, ep, freeze_d=False)
            val_g_loss, val_d_loss, val_acc = run_validation(model, criterion, val_dataloader)

            sample = generate_sample(model['g'])

            print("Epoch %d/%d\n"
                  "\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
                  "\t[Validation] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f" %
                  (ep+1, args.num_epochs, trn_g_loss, trn_d_loss, trn_acc,
                   val_g_loss, val_d_loss, val_acc))
            
            print(sample)


    for ep in range(args.num_epochs):
        model, trn_g_loss, trn_d_loss, trn_acc = run_training(model, optimizer, criterion, trn_dataloader, ep)
        val_g_loss, val_d_loss, val_acc = run_validation(model, criterion, val_dataloader)

        sample = generate_sample(model['g'])

        print("Epoch %d/%d\n"
              "\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
              "\t[Validation] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f" %
              (ep+1, args.num_epochs, trn_g_loss, trn_d_loss, trn_acc,
               val_g_loss, val_d_loss, val_acc))
        print(sample)

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
    ARG_PARSER.add_argument('--freeze_g', action='store_true')
    ARG_PARSER.add_argument('--freeze_d', action='store_true')
    ARG_PARSER.add_argument('--num_epochs', default=200, type=int)
    ARG_PARSER.add_argument('--seq_len', default=8, type=int)
    ARG_PARSER.add_argument('--batch_size', default=32, type=int)

    ARG_PARSER.add_argument('-m', action='store_true')
    ARG_PARSER.add_argument('--no_pretraining', action='store_true')
    ARG_PARSER.add_argument('--pretraining_epochs', default=10, type=int)

    ARGS = ARG_PARSER.parse_args()
    MAX_SEQ_LEN = ARGS.seq_len
    BATCH_SIZE = ARGS.batch_size
    FREEZE_G = ARGS.freeze_g
    FREEZE_D = ARGS.freeze_d

    main(ARGS)
