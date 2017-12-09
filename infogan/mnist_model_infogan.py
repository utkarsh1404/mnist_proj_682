
from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

import pre_train_net as pre


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

X_train_img, X_train_text, y_train, X_val_img, X_val_text, y_val, X_test_img, X_test_text, y_test = None, None, None, None, None, None, None, None, None
word2vec = np.load('/home/utkarsh1404/mnist_proj_682/word2vec_digits.npy').item()

samples_text = np.zeros((50,1,300))
gen_targets = []
for i in range(50):
    k = (i/5)
    #print (word2vec[str(k)].shape)
    samples_text[i] = np.reshape(np.array(word2vec[str(k)]),(1,300))
    gen_targets.append(k)

def create_word_vectors(l):
    ar = np.zeros((len(l),1,300))
    for id in range(len(l)):
        ar[id] = np.reshape(np.array(word2vec[str(l[id])]),(1,300))
    return ar  


def load_dataset():
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
    X_train, X_val = X_train[:tr_data_sz], None
    y_train, y_val = y_train[:tr_data_sz], None

    X_train_text = create_word_vectors(y_train)
    X_val_text = None# TODO : X_val_text = create_word_vectors(y_val)
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

def clip(x):
    return T.clip(x, 1e-8, 1-1e-8)

def build_generator(input_noise=None, input_c=None, input_text=None):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm, ConcatLayer
    from lasagne.nonlinearities import sigmoid
    # input: 100dim
    layer = InputLayer(shape=(None, noise_dim), input_var=input_noise)
    layer2 = InputLayer(shape=(None,1,300), input_var=input_text)
    layer2 = ReshapeLayer(layer2, ([0], 1*300))
    layer3 = InputLayer(shape=(None,2), input_var=input_c)


    layer = ConcatLayer([layer, layer2, layer3], axis=1)

    #increasing order of fc-layer
    for i in range(len(fclayer_list)):
        layer = batch_norm(DenseLayer(layer, fclayer_list[i]))
    
    newPS = 28
    if stride!=1:
        newPS = 28/(2**len(layer_list))

    layer = batch_norm(DenseLayer(layer, layer_list[0]*newPS*newPS))
    layer = ReshapeLayer(layer, ([0], layer_list[0], newPS, newPS))
    
    for i in range(1,len(layer_list)):
        layer = batch_norm(Deconv2DLayer(layer, layer_list[i], filter_sz, stride=stride, pad=(filter_sz-1)/2))
    layer = Deconv2DLayer(layer, 1, filter_sz, stride=stride, pad=(filter_sz-1)/2,
                          nonlinearity=sigmoid)
    print ("Generator output:", layer.output_shape)
    return layer

def build_discriminator(input_img=None, input_text=None):
    from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                DenseLayer, batch_norm, ConcatLayer)
    from lasagne.nonlinearities import LeakyRectify, sigmoid
    lrelu = LeakyRectify(0.1)
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 1, 28, 28), input_var=input_img)

    layer2 = InputLayer(shape=(None,1,300), input_var=input_text)
    layer2 = ReshapeLayer(layer2, ([0], 1*300))

    for i in reversed(range(len(layer_list))):
        layer = batch_norm(Conv2DLayer(layer, layer_list[i], filter_sz, stride=stride, pad=(filter_sz-1)/2, nonlinearity=lrelu)) 
       
    newPS = 28
    if stride!=1:
        newPS = 28/(2**len(layer_list))

    layer = ReshapeLayer(layer, ([0], layer_list[0]*newPS*newPS))
    layer = ConcatLayer([layer, layer2], axis=1)

    for i in reversed(range(len(fclayer_list))):
        layer = batch_norm(DenseLayer(layer, fclayer_list[i], nonlinearity=lrelu))
    
    layer_main = DenseLayer(layer, 1, nonlinearity=sigmoid)
    print ("Discriminator output 1:", layer_main.output_shape)


    layer = batch_norm(DenseLayer(layer , 128, nonlinearity=lrelu))
    l_Q_C_mean = DenseLayer(layer, 2, nonlinearity=linear)
    l_Q_C_stddev = DenseLayer(layer, 2, nonlinearity=T.exp)
    print ("Discriminator output 2:", l_Q_C_mean.output_shape, l_Q_C_stddev.output_shape)

    return layer_main, l_Q_C_mean, l_Q_C_stddev


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, text, batchsize, shuffle=False):
    assert len(inputs) == len(text)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], text[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def train_network(initial_eta):
    # Load the dataset
    print("Loading data...")
    X_train, X_train_text, y_train, X_val, X_val_text, y_val, X_test, X_test_text, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    c_var = T.dmatrix('c')
    noise_var = T.dmatrix('noise')
    input_img = T.dtensor4('inputs')
    input_text = T.dtensor3('text')

    # Create neural network model
    print("Building model and compiling functions...")
    generator = build_generator(noise_var, c_var, input_text)
    discriminator, c_mean, c_std = build_discriminator(input_img, input_text)

    all_layers = lasagne.layers.get_all_layers(discriminator)
    print ("LAYERS: ")
    print (all_layers)

    # Create expression for passing real data through the discriminator
    out_disc_main = lasagne.layers.get_output(discriminator)
    # Create expression for passing fake data through the discriminator
    out_gen_disc_main = lasagne.layers.get_output(discriminator,
            {all_layers[0]: lasagne.layers.get_output(generator), all_layers[2+3*len(layer_list)]: input_text})
    out_gen_disc_c_mean = lasagne.layers.get_output(c_mean,
            {all_layers[0]: lasagne.layers.get_output(generator), all_layers[2+3*len(layer_list)]: input_text})
    out_gen_disc_c_std = lasagne.layers.get_output(c_std,
            {all_layers[0]: lasagne.layers.get_output(generator), all_layers[2+3*len(layer_list)]: input_text})

    TINY = 1e-8
    # Create loss expressions
    loss_discriminator0 = -T.log(out_disc_main + TINY).mean() -  T.log(1. - out_gen_disc_main + TINY).mean()
    loss_generator0 = -T.log(out_gen_disc_main + TINY).mean()  

    epsilon = (c_var - out_gen_disc_c_mean) /(out_gen_disc_c_std + TINY)
    loss_Q_C = (T.log(out_gen_disc_c_std + TINY) + 0.5 * T.square(epsilon)).mean()  

    discriminator_loss = loss_discriminator0 + loss_Q_C
    generator_loss = loss_generator0 + loss_Q_C

    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params([discriminator,c_mean,c_std], trainable=True)
    #eta = theano.shared(lasagne.utils.floatX(initial_eta))
    updates_generator = lasagne.updates.adam(
            generator_loss, generator_params, learning_rate=1e-3, beta1=0.5)
    updates_discriminator = lasagne.updates.adam(
            discriminator_loss, discriminator_params, learning_rate=2e-4, beta1=0.5)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn_disc = theano.function([noise_var, c_var, input_img, input_text],
                                    [],
                                    updates=updates_discriminator)

    train_fn_gen = theano.function([noise_var, c_var, input_text],
                                    [],
                                    updates=updates_generator)

    # Compile another function generating some data
    gen_fn = theano.function([noise_var, c_var, input_text],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))
    
    ##TODO
    
    loss_func_calc = theano.function([noise_var, c_var, input_img, input_text],
                        [   -T.log(lasagne.layers.get_output(discriminator, deterministic=True) + TINY).mean() 
                          -T.log(1. - (lasagne.layers.get_output(discriminator,
                                    {all_layers[0]: lasagne.layers.get_output(generator, deterministic=True), all_layers[2+3*len(layer_list)]: input_text}
                                                                  ,deterministic=True)) 
                                        + TINY).mean() 
                          + (T.log(lasagne.layers.get_output(c_std, {all_layers[0]: lasagne.layers.get_output(generator,deterministic=True), all_layers[2+3*len(layer_list)]: input_text},deterministic=True)
                                    + TINY) + 0.5 * T.square((c_var - lasagne.layers.get_output(c_mean,
            {all_layers[0]: lasagne.layers.get_output(generator,deterministic=True), all_layers[2+3*len(layer_list)]: input_text},deterministic=True)) / (lasagne.layers.get_output(c_std,
                                    {all_layers[0]: lasagne.layers.get_output(generator,deterministic=True), all_layers[2+3*len(layer_list)]: input_text},deterministic=True) + TINY))).mean(),
                         ##])
    

    get_acc = theano.function([noise_var, c_var, input_img, input_text],
                              [(lasagne.layers.get_output(discriminator, deterministic=True) > .5).mean(),
                               (lasagne.layers.get_output(discriminator, {all_layers[0] : lasagne.layers.get_output(generator, deterministic=True), all_layers[2+3*len(layer_list)] : input_text}, deterministic=True) < .5).mean()])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_acc_d = 0
        train_acc_g = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, X_train_text, batch_sz, shuffle=True):
            inputs, text = batch
            noise = lasagne.utils.floatX(np.random.uniform(low=-1, high=1, size=(len(inputs), noise_dim)))
            value_c = lasagne.utils.floatX(np.random.uniform(low=-1, high=1, size=(len(inputs), 2)))
            ##TODO : check return type
            train_acc_d += np.array(train_fn_disc(noise, value_c, inputs, text))
            train_acc_g += np.array(train_fn_gen(noise, value_c, text))
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print(" disc (R/F) training acc (avg in an epoch):\t\t{}".format(train_acc_d / train_batches))

        #loss
        new_noise = lasagne.utils.floatX(np.random.uniform(low=-1, high=1, size=(X_train.shape[0], noise_dim)))
        new_value_c = lasagne.utils.floatX(np.random.uniform(low=-1, high=1, size=(X_train.shape[0], 2)))
        loss_val = loss_func_calc(new_noise, new_value_c, X_train, X_train_text)
        acc_val = get_acc(new_noise, new_value_c, X_train, X_train_text)
        print("DISC/GEN LOSS VALUE AT EPOCH : ", epoch+1, " = ", loss_val)
        print("DISC (R/F) ACC VALUE AT EPOCH : ", epoch+1, " = ", acc_val)

        # And finally, we plot some generated data
        if epoch%2==0:
            new_noise = lasagne.utils.floatX(np.random.uniform(low=-1, high=1, size=(50, noise_dim)))
            new_value_c = 0.5 * np.ones((50,2))
            samples = gen_fn(new_noise, new_value_c, samples_text)
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                pass
            else:
                plt.imsave(run+'/mnist_samples_'+str(epoch)+'.png',
                   		(samples.reshape(5, 10, 28, 28)
                           .transpose(0, 2, 1, 3)
                           .reshape(5*28, 10*28)),
                   		cmap='gray')
                curr_epoch_pred = pre.make_predictions(samples, gen_targets)
                print ("In this epoch = ", epoch+1, " : my generated sample pretrained acc is : ", curr_epoch_pred)

                acc_val_sample = get_acc(new_noise, new_value_c, X_train[:50], samples_text)
                print ("in this epoch = ", epoch+1, " : my generated samples in the discrimantor being predicted as real had accuracy : ", 1-acc_val_sample[1])

                kl_divergence, mode_score = pre.findInceptionScore(samples, gen_targets)
                print ("in this epoch = ", epoch+1, " : my generated samples had inception score : ", kl_divergence, " ; ", mode_score)

        # After half the epochs, we start decaying the learn rate towards zero
        '''
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))
        '''
    # Optionally, you could now dump the network weights to a file like this:
    np.savez(run+'/mnist_gen.npz', *lasagne.layers.get_all_param_values(generator))
    np.savez(run+'/mnist_disc.npz', *lasagne.layers.get_all_param_values([discriminator, c_mean, c_var]))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


layer_list = None
fclayer_list = None
tr_data_sz = None
noise_dim = None
filter_sz = None
stride = None
num_epochs=None
loss_func=None
batch_sz = None

run = None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tr_data_size', required=False, type=int, default=10000)
    parser.add_argument('--noise_dim', required=False, type=int, default=100)
    parser.add_argument('--filter_sz', required=False, type=int, default=5)#odd
    parser.add_argument('--stride', required=False, type=int, default=2)
    parser.add_argument('--num_epochs', required=False, type=int, default=10)
    parser.add_argument('--loss_func', required=False, type=int, default=0)
    parser.add_argument('--lr', required=False, type=float, default=2e-4)
    parser.add_argument('--batch_size', required=False, type=int, default=128)
    parser.add_argument('--layer_list', nargs='+', type=int, default=[128,64])
    parser.add_argument('--fclayer_list', nargs='+', type=int, default=[1024])

    parser.add_argument('--run', required=True, type=str, default=0)

    args = parser.parse_args()

    tr_data_sz = args.tr_data_size
    noise_dim = args.noise_dim
    filter_sz = args.filter_sz
    stride = args.stride
    num_epochs=args.num_epochs
    loss_func=args.loss_func
    batch_sz = args.batch_size
    run = args.run

    layer_list = args.layer_list
    fclayer_list = args.fclayer_list

    lr = args.lr

    if not os.path.exists(run):
        os.makedirs(run)

    train_network(lr)

