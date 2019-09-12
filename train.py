''' Train C-RNN-GAN
'''
import torch
import torch.nn as nn
from torch import optim

from c_rnn_gan import Generator, Discriminator
import music_data_utils
from midi_statistics import get_all_stats

DATADIR = 'data'
TRAINDIR = 'traindump'
LRN_RATE = 0.1
D_LRN_FACTOR = 0.5
LRN_DECAY = 1.0
EPOCHS_BEFORE_DECAY = 60
MAX_GRAD_NORM = 5.0
BATCH_SIZE = 16
MAX_EPOCHS = 500

COMPOSER = 'mozart'
MAX_SEQ_LEN = 100

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


def run_epoch(model, optimizer, criterion, dataloader, mode='train'):
    ''' Run single epoch
    '''
    loss = {
        'g': 10.0,
        'd': 10.0
    }

    num_feats = dataloader.get_num_song_features()
    dataloader.rewind(part=mode)
    batch_meta, batch_song = dataloader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part=mode)
    while batch_meta is not None and batch_song is not None:

        # loss checking
        if mode == 'train':
            if not check_loss(model, loss):
                break

        # get initial states
        g_states = model['g'].init_hidden(BATCH_SIZE)
        d_state = model['d'].init_hidden(BATCH_SIZE)

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        # h = tuple([each.data for each in h])

        #### GENERATOR ####
        optimizer['g'].zero_grad()
        # prepare inputs
        z = torch.empty([BATCH_SIZE, MAX_SEQ_LEN, num_feats]).uniform_() # random vector
        batch_song = torch.Tensor(batch_song)

        # feed inputs to generator
        g_feats, g_states = model['g'](z, g_states)
        # feed real and generated input to discriminator
        _, d_feats_real, _ = model['d'](batch_song, d_state)
        _, d_feats_gen, _ = model['d'](g_feats, d_state)
        # calculate loss, backprop, and update weights of G
        loss['g'] = criterion['g'](d_feats_real, d_feats_gen)
        loss['g'].backward()
        nn.utils.clip_grad_norm_(model['g'].parameters(), max_norm=MAX_GRAD_NORM)
        optimizer['g'].step()

        #### DISCRIMINATOR ####
        optimizer['d'].zero_grad()
        # feed real and generated input to discriminator
        d_logits_real, _, _ = model['d'](batch_song, d_state)
        # need to detach from operation history to prevent backpropagating to generator
        d_logits_gen, _, _ = model['d'](g_feats.detach(), d_state)
        # calculate loss, backprop, and update weights of D
        loss['d'] = criterion['d'](d_logits_real, d_logits_gen)
        loss['d'].backward()
        nn.utils.clip_grad_norm_(model['d'].parameters(), max_norm=MAX_GRAD_NORM)
        optimizer['d'].step()

        # fetch next batch
        batch_meta, batch_song = dataloader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part=mode)

        import sys
        sys.exit()
                

def train():
    ''' Training sequence
    '''
    dataloader = music_data_utils.MusicDataLoader(DATADIR, single_composer=COMPOSER)
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
        'g': optim.SGD(model['g'].parameters(), LRN_RATE, weight_decay=1.0),
        'd': optim.SGD(model['d'].parameters(), LRN_RATE*D_LRN_FACTOR, weight_decay=1.0)
    }

    criterion = {
        'g': nn.MSELoss(), # feature matching
        'd': DLoss()
    }

    if(train_on_gpu):
        model['g'].cuda()
        model['d'].cuda()

    for ep in range(MAX_EPOCHS):
        run_epoch(model, optimizer, criterion, dataloader, mode='train')

        # validation
        # g_model.eval()
        # model['d'].eval()



if __name__ == "__main__":
    train()