# Music Generation using C-RNN-GAN in PyTorch

## Status
* This is a **work in progress**. I'm still meddling with the MIDI dataset (which I acquired from the original C-RNN-GAN repository).
* **As far as implementation of the C-RNN-GAN architecture is concerned, code is already finished.** I have yet to train a generator that can output some pleasant MIDI though :sweat_smile:. I'll upload the model and the generated MIDI once done.
* I provided a [sample code](train_simple.py) which uses a number sequence as dataset. I used this to validate the architecture implemnetation. Using this code, the C-RNN-GAN generator can fairly mimic the input number sequence after 100+ training epochs.

## Prerequisites
* Python 3.6
* PyTorch
* [python3-midi](https://github.com/louisabraham/python3-midi)

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details


**Original Implementation (TensorFlow)**: https://github.com/olofmogren/c-rnn-gan

## Things to Try

* CHECK VALIDITY OF BATCHES!!!

* Need to find hyperparams that match
	* batch size & pre-training epochs
	* batch size & learning rate

* Label smoothing (turn positive label from 1 to a value like 0.9) !
* Multi-scale gradient (https://arxiv.org/abs/1903.06048) - possible for C-RNN-GAN?
* TTUR (D lr > G lr)
* Spectral normalization (https://arxiv.org/abs/1802.05957) - possible for C-RNN-GAN?

* Try not to use feature matching
* "No wasted midi" - fix batches
* SGD w/ decay
* Min-max all features then use sigmoid


## Overfit Test

Trial 1: default params, feature_matching, label smoothing, 100 epochs
