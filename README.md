# Music Generation using C-RNN-GAN in PyTorch

## Introduction

This project is a PyTorch implementation of [C-RNN-GAN](https://github.com/olofmogren/c-rnn-gan), which was originally developed in TensorFlow. In a nutshell, C-RNN-GAN is a GAN variant where both the Generator and the Discriminator are RNNs, with each output at each timestep from the Generator are correspondingly fed into each timestep as input to the Discriminator. The goal is to train the Generator to output structured sequences, such as MIDI music which was used in the paper. If you'd like to know more, head over to this [link](http://mogren.one/publications/2016/c-rnn-gan/) or read the [paper](http://mogren.one/publications/2016/c-rnn-gan/mogren2016crnngan.pdf).

## Status

The implementation can work well on simple sequences such as `a(n+1) = 2*a(n)`, where each element is twice of the previous. You can try this by executing:
```
$ python train_simple.py
```
This runs for 200 epochs, after which you should get something similar to this:

![Simple output](images/simple_out.png)

`TBD: status on MIDI`

## Prerequisites
* Python 3.6
* PyTorch
* [python3-midi](https://github.com/louisabraham/python3-midi)

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details
