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
    ''' Freeze/unfreeze optimization
    '''
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    else: # unfreeze
        for param in model.parameters():
            param.requires_grad = True 


def run_epoch(mode='train'):
    ''' Run single epoch
    '''
    g_loss, d_loss = 10.0, 10.0
    loader.rewind(part=mode)

    _, batch_song = loader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part=mode)
    while batch_meta is not None and batch_song is not None:
        control_grad(g_model, freeze=False)
        control_grad(d_model, freeze=False)

        # loss checking
        if mode == 'train':
            if d_loss == 0.0 and g_loss == 0.0:
                print('Both G and D train loss are zero. Exiting.')
                break
            elif d_loss == 0.0: # freeze D
                control_grad(d_model, freeze=True)
            elif g_loss == 0.0: # freeze G
                control_grad(g_model, freeze=True)
            elif g_loss < 2.0 or d_loss < 2.0:
                control_grad(d_model, freeze=True)
                if g_loss*0.7 > d_loss:
                    control_grad(g_model, freeze=True)

        # NOTE handling of states!

        
        # feed real and generated data to discriminator
        batch_song = torch.Tensor(batch_song)
        d_logits_real, d_feats_real, d_state = d_model(batch_song, d_state)
        
        d_logits_gen, d_feats_gen, d_state = d_model(g_feats.detach(), d_state)


        # fetch next batch
        _, batch_song = loader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part=mode)
                


def train():
    ''' Training sequence
    '''
    loader = music_data_utils.MusicDataLoader(DATADIR, single_composer=COMPOSER)
    num_feats = loader.get_num_song_features()

    import sys
    sys.exit()

    # First checking if GPU is available
    train_on_gpu=torch.cuda.is_available()
    if(train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    g_model = Generator(num_feats)
    d_model = Discriminator(num_feats)

    g_optimizer = optim.SGD(g_model.parameters(), LRN_RATE, weight_decay=1.0)
    d_optimizer = optim.SGD(d_model.parameters(), LRN_RATE*D_LRN_FACTOR, weight_decay=1.0)

    g_criterion = nn.MSELoss() # feature matching
    d_criterion = DLoss()

    if(train_on_gpu):
        g_model.cuda()
        d_model.cuda()

    for ep in range(MAX_EPOCHS):
        lr_decay = LRN_DECAY ** max(ep - EPOCHS_BEFORE_DECAY, 0.0)

        #### GENERATOR ####
        g_optimizer.zero_grad()
        # generate random vector
        z = torch.empty([BATCH_SIZE, MAX_SEQ_LEN, num_feats]).uniform_()
        # feed inputs to generator
        g_feats, g_states = g_model(z, g_states)
        # feed real and generated input to discriminator
        _, d_feats_real, d_state = d_model(batch_song, d_state)
        _, d_feats_gen, d_state = d_model(g_feats, d_state)
        # calculate loss, backprop, and update weights of G
        g_loss = g_criterion(d_feats_real, d_feats_gen)
        g_loss.backward()
        nn.utils.clip_grad_norm(g_model.parameters(), max_norm=MAX_GRAD_NORM)
        g_optimizer.step()

        #### DISCRIMINATOR ####
        d_optimizer.zero_grad()
        # feed real and generated input to discriminator
        d_logits_real, _, d_state = d_model(batch_song, d_state)
        # need to detach from operation history to prevent backpropagating to generator
        d_logits_gen, _, d_state = d_model(g_feats.detach(), d_state)
        # calculate loss, backprop, and update weights of D
        d_loss = d_criterion(d_logits_real, d_logits_gen)
        d_loss.backward()
        nn.utils.clip_grad_norm(d_model.parameters(), max_norm=MAX_GRAD_NORM)
        d_optimizer.step()

        # validation
        g_model.eval()
        d_model.eval()



if __name__ == "__main__":
    train()