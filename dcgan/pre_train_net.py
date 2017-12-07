from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

import math

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

# Create neural network model
print("LOADED PRETRAINED...")
network = build_cnn(input_var)

with np.load('model.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(network, param_values)
test_prediction = lasagne.layers.get_output(network, deterministic=True)


predict = theano.function([input_var], test_prediction)
#print (y[1])
#print (np.array((train_fn(np.reshape(x[1],(1,1,28,28))))))


def make_predictions(samples, targets):
    vals = predict(samples)
    acc_val = 0.0
    for n in range(len(targets)):
        pred = np.argmax(np.array(vals[n]))
        actual = targets[n]
        if pred==actual:
            acc_val+=1.0
    return acc_val/len(targets)


def findInceptionScore(samples, targets):
    vals = predict(samples)
    kl = 0.0
    for n in range(samples.shape[0]):
        c_val = 0.0
        pred = np.array(vals[n])
        for elem in pred:
            c_val += elem * math.log(elem*10)
        kl += c_val       
    return kl/samples.shape[0]
