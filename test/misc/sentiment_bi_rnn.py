import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np

import os
from string import punctuation
from collections import Counter

# IMPORTANT: change this to where 'data' folder is located
#HOME = 'deep-learning-v2-pytorch/sentiment-rnn'
HOME = '/home/cjbayron/machine_learning/pytorch/deep-learning-v2-pytorch/sentiment-rnn'

def get_path(path):
	return os.path.join(HOME, path)

def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    ## implement function
    
    features=[]
    for review in reviews_ints:
        if len(review) < seq_length:
            features.append(([0] * (seq_length - len(review))) + review)
        elif len(review) > seq_length:
            features.append(review[:seq_length])
        else:
            features.append(review)
    
    features = np.array(features)
    return features

def get_trn_val_tst_split(split_frac, idxs):
	
	np.random.shuffle(idxs)
	train_idxs = idxs[: int(split_frac * len(idxs))]

	rem = idxs[int(split_frac * len(idxs)) : ]
	val_idxs, test_idxs = rem[: int(0.5 * len(rem))], rem[int(0.5 * len(rem)) : ]

	return train_idxs, val_idxs, test_idxs


def get_loader(feats, labels, batch_size):
	data = TensorDataset(torch.from_numpy(feats), torch.from_numpy(labels))
	return DataLoader(data, shuffle=True, batch_size=batch_size)

def predict(net, test_review, sequence_length=200):
    ''' Prints out whether a give review is predicted to be 
        positive or negative in sentiment, using a trained model.
        
        params:
        net - A trained net 
        test_review - a review made of normal text and punctuation
        sequence_length - the padded length of a review
        '''
    
    test_review = test_review.lower() # lowercase, standardize
    all_text = ''.join([c for c in test_review if c not in punctuation])

    test_review = all_text.split()
    test_review = pad_features([test_review], sequence_length)
    test_review = test_review[0]
    
    word_list = []
    for word in test_review:
        if word in vocab_to_int:
            word_list.append( vocab_to_int[word] )
        else:
            word_list.append(0)
    
    test_review_int = torch.LongTensor(word_list)
    test_review_int.unsqueeze_(0)
    
    net.eval()
    val_h = net.init_hidden(1)
    #print(test_review_int.shape)
    if(train_on_gpu):
        test_review_int = test_review_int.cuda()
        
    out, h = net.forward(test_review_int, val_h)
    
    # print custom response based on whether test_review is pos/neg
    if out >= 0.5:
        print("Positive!")
    else:
        print("Negative!")

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, seq_len, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.bidirectional=True
        
        # define all layers
        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim,
                                      padding_idx=None)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=n_layers, batch_first=True, dropout=drop_prob,
                            bidirectional=self.bidirectional)
        
        self.dropout = nn.Dropout(drop_prob)
        self.output_layer = nn.Linear(in_features=hidden_dim, out_features=output_size)
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        emb_x = self.emb_layer(x)
        lstm_out, hidden = self.lstm(emb_x, hidden)
        if self.bidirectional:
            # separate to forward and backward
            lstm_out = lstm_out.contiguous().view(-1, self.seq_len, 2, self.hidden_dim)
            # get backward output in first
            # get forward output in last
            lstm_out_bw = lstm_out[:, 0, 1, :]
            lstm_out_fw = lstm_out[:, -1, 0, :]
            lstm_out = torch.add(input=lstm_out_bw, alpha=1, other=lstm_out_fw)
            lstm_out = torch.div(lstm_out, 2)
        else:
            lstm_out = lstm_out[:, -1]
        
        assert lstm_out.shape[-1] == self.hidden_dim, (lstm_out.shape, self.hidden_dim)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        d_out = self.dropout(lstm_out)
        fc_out = self.output_layer(d_out)
        sig_out = torch.sigmoid(fc_out)
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        layer_mult = 1
        if self.bidirectional:
            layer_mult *= 2
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers * layer_mult, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers * layer_mult, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers * layer_mult, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers * layer_mult, batch_size, self.hidden_dim).zero_())
        
        return hidden

if __name__ == "__main__":

	#####################
	# DATA PREPROCESSING
	#####################

	# read data from text files
	with open(get_path('data/reviews.txt'), 'r') as f:
	    reviews = f.read()
	with open(get_path('data/labels.txt'), 'r') as f:
	    labels = f.read()

	# get rid of punctuation
	reviews = reviews.lower() # lowercase, standardize
	all_text = ''.join([c for c in reviews if c not in punctuation])

	# split by new lines and spaces
	reviews_split = all_text.strip().split('\n')
	all_text = ' '.join(reviews_split)

	# create a list of words
	words = all_text.split()

	## Build a dictionary that maps words to integers
	vocab_to_int = { word:(index+1) for index, (word, freq) in enumerate(Counter(words).most_common()) }

	## use the dict to tokenize each review in reviews_split
	## store the tokenized reviews in reviews_ints
	reviews_ints = []
	for review in reviews_split:
	    reviews_ints.append([ vocab_to_int[word] for word in review.split() ])

	# 1=positive, 0=negative label conversion
	labels_list = labels.strip().split('\n')
	label_map = {'positive':1, 'negative':0}
	encoded_labels = [ label_map[str_label] for str_label in labels_list ]

	assert len(encoded_labels) == 25000
	assert len(reviews_ints) == 25000

	# Test your implementation!
	seq_length = 200
	features = pad_features(reviews_ints, seq_length=seq_length)
	#print(len(features), len(reviews_ints))
	## test statements - do not change - ##
	assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
	assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

	idxs = np.arange(len(features))
	train_idxs, val_idxs, test_idxs = get_trn_val_tst_split(0.8, idxs)

	# convert to np array
	labels = np.array(encoded_labels)

	get_split = lambda i : (features[i], labels[i])
	trn_feat, trn_labels = get_split(train_idxs)
	val_feat, val_labels = get_split(val_idxs)
	tst_feat, tst_labels = get_split(test_idxs)

	batch_size = 50
	train_loader = get_loader(trn_feat, trn_labels, batch_size)
	valid_loader = get_loader(val_feat, val_labels, batch_size)
	test_loader = get_loader(tst_feat, tst_labels, batch_size)

	# Instantiate the model w/ hyperparams
	vocab_size = len(vocab_to_int) + 1
	output_size = 1
	embedding_dim = 128
	hidden_dim = 128
	n_layers = 2

	net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, seq_length, drop_prob=0.6)

	###########
	# TRAINING
	###########

	# First checking if GPU is available
	train_on_gpu=torch.cuda.is_available()

	if(train_on_gpu):
	    print('Training on GPU.')
	else:
	    print('No GPU available, training on CPU.')

	# loss and optimization functions
	lr=0.001

	criterion = nn.BCELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)

	# training params

	epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing

	counter = 0
	print_every = 100
	clip=5 # gradient clipping

	# move model to GPU, if available
	if(train_on_gpu):
	    net.cuda()

	net.train()
	# train for some number of epochs
	for e in range(epochs):
	    # initialize hidden state
	    h = net.init_hidden(batch_size)

	    # batch loop
	    for inputs, labels in train_loader:
	        counter += 1

	        if(train_on_gpu):
	            inputs, labels = inputs.cuda(), labels.cuda()

	        # Creating new variables for the hidden state, otherwise
	        # we'd backprop through the entire training history
	        h = tuple([each.data for each in h])

	        # zero accumulated gradients
	        net.zero_grad()

	        # get the output from the model
	        output, h = net(inputs, h)

	        # calculate the loss and perform backprop
	        loss = criterion(output.squeeze(), labels.float())
	        loss.backward()
	        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
	        nn.utils.clip_grad_norm_(net.parameters(), clip)
	        optimizer.step()

	        # loss stats
	        if counter % print_every == 0:
	            # Get validation loss
	            val_h = net.init_hidden(batch_size)
	            val_losses = []
	            net.eval()
	            for inputs, labels in valid_loader:

	                # Creating new variables for the hidden state, otherwise
	                # we'd backprop through the entire training history
	                val_h = tuple([each.data for each in val_h])

	                if(train_on_gpu):
	                    inputs, labels = inputs.cuda(), labels.cuda()

	                output, val_h = net(inputs, val_h)
	                val_loss = criterion(output.squeeze(), labels.float())

	                val_losses.append(val_loss.item())

	            net.train()
	            print("Epoch: {}/{}...".format(e+1, epochs),
	                  "Step: {}...".format(counter),
	                  "Loss: {:.6f}...".format(loss.item()),
	                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

	###########
	# TESTING
	###########

	# Get test data loss and accuracy
	test_losses = [] # track loss
	num_correct = 0

	# init hidden state
	h = net.init_hidden(batch_size)

	net.eval()
	# iterate over test data
	for inputs, labels in test_loader:

	    # Creating new variables for the hidden state, otherwise
	    # we'd backprop through the entire training history
	    h = tuple([each.data for each in h])

	    if(train_on_gpu):
	        inputs, labels = inputs.cuda(), labels.cuda()
	    
	    # get predicted outputs
	    output, h = net(inputs, h)
	    
	    # calculate loss
	    test_loss = criterion(output.squeeze(), labels.float())
	    test_losses.append(test_loss.item())
	    
	    # convert output probabilities to predicted class (0 or 1)
	    pred = torch.round(output.squeeze())  # rounds to the nearest integer
	    
	    # compare predictions to true label
	    correct_tensor = pred.eq(labels.float().view_as(pred))
	    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
	    num_correct += np.sum(correct)


	# -- stats! -- ##
	# avg test loss
	print("Test loss: {:.3f}".format(np.mean(test_losses)))
	# accuracy over all test data
	test_acc = num_correct/len(test_loader.dataset)
	print("Test accuracy: {:.3f}".format(test_acc))


	#############
	# PREDICTION
	#############
	
	# negative test review
	test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'
	# positive test review
	test_review_pos = 'This movie had the best acting and the dialogue was so good. I loved it.'
	# call function
	# try negative and positive reviews!
	seq_length=200
	predict(net, test_review_neg, seq_length)
	predict(net, test_review_pos, seq_length)