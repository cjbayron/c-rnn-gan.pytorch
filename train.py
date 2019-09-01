import pandas as pd
from string import punctuation
from string import ascii_lowercase

import c_rnn_gan

def char_filter(char):
	nums = [str(i) for i in range(0,10)]
	return ( (char in punctuation)
		      or (char in nums)
		      or (char in ascii_lowercase)
		      or char == ' ' )

if __name__ == "__main__":
	# load data
	quotes_df = pd.read_json('assets/quotes.json')
	# get only inspirational quotes
	quotes_df = quotes_df[ quotes_df['Category'] == 'life' ]
	# get string quotes
	quotes = ' '.join(quotes_df['Quote'].values.tolist()).lower()
	quotes = list(filter(char_filter, quotes))

	batch_size = 64
	seq_len = 64

	