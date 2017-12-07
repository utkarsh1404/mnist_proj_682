
from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.


word2vec = np.load('/home/kruti/text2image/data/digits/word2vec_digits.npy').item()

samples_text = np.zeros((50,1,300))
for i in range(50):
    k = (i/5)%10
    print (word2vec[str(k)].shape)
    samples_text[i] = np.reshape(np.array(word2vec[str(k)]),(1,300))

def create_word_vectors(l):
    ar = np.zeros((len(l),1,300))
    for id in range(len(l)):
        ar[id] = np.reshape(np.array(word2vec[str(l[id])]),(1,300))
    return ar  


def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:10000], X_train[-10000:]
    y_train, y_val = y_train[:10000], y_train[-10000:]

    X_train_text = create_word_vectors(y_train)
    X_val_text = create_word_vectors(y_val)
    X_test_text= create_word_vectors(y_test)

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, X_train_text, y_train, X_val, X_val_text, y_val, X_test, X_test_text, y_test


# ##################### Build the neural network model #######################
# We create two models: The generator and the discriminator network. The
# generator needs a transposed convolution layer defined first.

class Deconv2DLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
            nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.W = self.add_param(lasagne.init.Orthogonal(),
                (self.input_shape[1], num_filters) + self.filter_size,
                name='W')
        self.b = self.add_param(lasagne.init.Constant(0),
                (num_filters,),
                name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        shape = tuple(i*s - 2*p + f - 1
                for i, s, p, f in zip(input_shape[2:],
                                      self.stride,
                                      self.pad,
                                      self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
            subsample=self.stride, border_mode=self.pad)
        conved = op(self.W, input, self.output_shape[2:])
        if self.b is not None:
            conved += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(conved)

def build_generator(input_noise=None, input_text=None):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm, ConcatLayer
    from lasagne.nonlinearities import sigmoid
    # input: 100dim
    layer = InputLayer(shape=(None, 100), input_var=input_noise)
    layer2 = InputLayer(shape=(None,1,300), input_var=input_text)
    layer2 = ReshapeLayer(layer2, ([0], 1*300))


    layer = ConcatLayer([layer, layer2], axis=1)
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 450))
    layer = batch_norm(DenseLayer(layer, 500))
    layer = batch_norm(DenseLayer(layer, 575))
    layer = batch_norm(DenseLayer(layer, 625))
    layer = batch_norm(DenseLayer(layer, 725))
    layer = batch_norm(DenseLayer(layer, 1*28*28, nonlinearity=sigmoid))
    layer = ReshapeLayer(layer, ([0], 1, 28, 28))
    print ("Generator output:", layer.output_shape)
    return layer

def build_discriminator(input_img=None, input_text=None):
    from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                DenseLayer, batch_norm, ConcatLayer)
    from lasagne.nonlinearities import LeakyRectify, sigmoid
    lrelu = LeakyRectify(0.2)
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 1, 28, 28), input_var=input_img)
    layer = ReshapeLayer(layer, ([0], 1*28*28))
    
    layer2 = InputLayer(shape=(None,1,300), input_var=input_text)
    layer2 = ReshapeLayer(layer2, ([0], 1*300))

    layer = ConcatLayer([layer, layer2], axis=1)

    layer = batch_norm(DenseLayer(layer, 725, nonlinearity=lrelu))
    layer = batch_norm(DenseLayer(layer, 625, nonlinearity=lrelu))
    layer = batch_norm(DenseLayer(layer, 575, nonlinearity=lrelu))
    layer = batch_norm(DenseLayer(layer, 500, nonlinearity=lrelu))
    layer = batch_norm(DenseLayer(layer, 450, nonlinearity=lrelu))
    # output layer
    layer = DenseLayer(layer, 1, nonlinearity=sigmoid)
    print ("Discriminator output:", layer.output_shape)
    return layer
    

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, text, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], text[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=200, loss_func=0, initial_eta=2e-4):
    # Load the dataset
    print("Loading data...")
    X_train, X_train_text, y_train, X_val, X_val_text, y_val, X_test, X_test_text, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    noise_var = T.dmatrix('noise')
    input_img = T.dtensor4('inputs')
    input_text = T.dtensor3('text')
#    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    generator = build_generator(noise_var, input_text)
    discriminator = build_discriminator(input_img, input_text)

    all_layers = lasagne.layers.get_all_layers(discriminator)
    print ("LAYERS: ")
    print (all_layers)

    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator)
    # Create expression for passing fake data through the discriminator
    fake_out = lasagne.layers.get_output(discriminator,
            {all_layers[0]: lasagne.layers.get_output(generator), all_layers[2]: input_text})

    # Create loss expressions
    generator_loss = None
    discriminator_loss = None

    if loss_func==0:
        generator_loss = lasagne.objectives.squared_error(fake_out, 1).mean()
        discriminator_loss = (lasagne.objectives.squared_error(real_out, 1) + lasagne.objectives.squared_error(fake_out, 0)).mean()
    else:
        generator_loss = lasagne.objectives.binary_crossentropy(fake_out, 1).mean()
        discriminator_loss = (lasagne.objectives.binary_crossentropy(real_out, 1) + lasagne.objectives.binary_crossentropy(fake_out, 0)).mean()
    
    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(initial_eta))
    updates = lasagne.updates.adam(
            generator_loss, generator_params, learning_rate=eta, beta1=0.5)
    updates.update(lasagne.updates.adam(
            discriminator_loss, discriminator_params, learning_rate=eta, beta1=0.5))

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([noise_var, input_img, input_text],
                               [(real_out > .5).mean(),
                                (fake_out < .5).mean()],
                               updates=updates)

    # Compile another function generating some data
    gen_fn = theano.function([noise_var, input_text],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, X_train_text, y_train, 128, shuffle=True):
            inputs, text, targets = batch
            noise = lasagne.utils.floatX(np.random.rand(len(inputs), 100))
            train_err += np.array(train_fn(noise, inputs, text))
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{}".format(train_err / train_batches))

        # And finally, we plot some generated data
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(50, 100)), samples_text)
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pass
        else:
            plt.imsave('mnist_samples.png',
                       (samples.reshape(5, 10, 28, 28)
                               .transpose(0, 2, 1, 3)
                               .reshape(5*28, 10*28)),
                       cmap='gray')

        # After half the epochs, we start decaying the learn rate towards zero
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))

    # Optionally, you could now dump the network weights to a file like this:
    #np.savez('mnist_gen.npz', *lasagne.layers.get_all_param_values(generator))
    #np.savez('mnist_disc.npz', *lasagne.layers.get_all_param_values(discriminator))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a DCGAN on MNIST using Lasagne.")
        print("Usage: %s [EPOCHS]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 100)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
            kwargs['loss_func'] = int(sys.argv[2])
main(**kwargs)
