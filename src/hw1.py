"""
All tensorflow objects, if not otherwise specified, should be explicity
created with tf.float32 datatypes. Not specifying this datatype for variables and
placeholders will cause your code to fail some tests.

For the specified functionality in this assignment, there are generally high
level Tensorflow library calls that can be used. As we are assessing tensorflow,
functionality that is technically correct but implemented manually, using a
library such as numpy, will fail tests. If you find yourself writing 50+ line
methods, it may be a good idea to look for a simpler solution.

Along with the provided functional prototypes, there is another file,
"train.py" which calls the functions listed in this file. It trains the
specified network on the MNIST dataset, and then optimizes the loss using a
standard gradient decent optimizer. You can run this code to check the models
you create.

"""

import tensorflow as tf
import math

def input_placeholder():
    """
    This placeholder serves as the input to the model, and will be populated
    with the raw images, flattened into single row vectors of length 784.

    The number of images to be stored in the placeholder for each minibatch,
    i.e. the minibatch size, may vary during training and testing, so your
    placeholder must allow for a varying number of rows.

    :return: A tensorflow placeholder of type float32 and correct shape
    """
    return tf.placeholder(dtype=tf.float32, shape=[None, 784],
                          name="image_input")

def target_placeholder():
    """
    This placeholder serves as the output for the model, and will be
    populated with targets for training, and testing. Each output will
    be a single one-hot row vector, of length equal to the number of
    classes to be classified (hint: there's one class for each digit)

    The number of target rows to be stored in the placeholder for each
    minibatch, i.e. the minibatch size, may vary during training and
    testing, so your placeholder must allow for a varying number of
    rows.

    :return: A tensorflow placeholder of type float32 and correct shape
    """
    return tf.placeholder(dtype=tf.float32, shape=[None, 10],
                          name="image_target_onehot")

def onelayer(X, Y, layersize=10,INPUT_SIZE=784,OUTPUT_SIZE=10):
    """
    Create a Tensorflow model for logistic regression (i.e. single layer NN)

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned  (variables in the
    python sense, not in the Tensorflow sense, although some may be
    Tensorflow variables). They must be returned in the following order.
        w: Connection weights
        b: Biases
        logits: The input to the activation function
        preds: The output of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    with tf.name_scope('layer1'):
        w = tf.Variable(
            tf.truncated_normal([INPUT_SIZE, layersize],
                                stddev=1.0 / math.sqrt(float(INPUT_SIZE))),
            name='weights')
        b = tf.Variable(tf.zeros([layersize]),
                             name='biases')
        hidden = tf.nn.relu(tf.matmul(X, w) + b)

    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([layersize, OUTPUT_SIZE],
                                stddev=1.0 / math.sqrt(float(layersize))),
            name='weights')
        biases = tf.Variable(tf.zeros([OUTPUT_SIZE]),
                             name='biases')
        logits = tf.matmul(hidden, weights) + biases

    preds=tf.nn.softmax(logits)
    batch_xentropy =  tf.nn.softmax_cross_entropy_with_logits(labels=Y,
                                                  logits=logits,
                                                  name="xentropy")

    batch_loss=tf.reduce_mean(batch_xentropy)



    return w, b, logits, preds, batch_xentropy, batch_loss

def twolayer(X, Y, hiddensize1=100, hiddensize2=30,INPUT_SIZE=784,OUTPUT_SIZE=10):
    """
    Create a Tensorflow model for a Neural Network with one hidden layer

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        W1: Connection weights for the first layer
        b1: Biases for the first layer
        W2: Connection weights for the second layer
        b2: Biases for the second layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    with tf.name_scope('layer1'):
        w1 = tf.Variable(
            tf.truncated_normal([INPUT_SIZE, hiddensize1],
                                stddev=1.0 / math.sqrt(float(INPUT_SIZE))),
            name='weights')
        b1 = tf.Variable(tf.zeros([hiddensize1]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(X, w1) + b1)
        # Hidden 2
    with tf.name_scope('layer2'):
        w2 = tf.Variable(
            tf.truncated_normal([hiddensize1, hiddensize2],
                                stddev=1.0 / math.sqrt(float(hiddensize1))),
            name='weights')
        b2 = tf.Variable(tf.zeros([hiddensize2]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)
        # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hiddensize2, OUTPUT_SIZE],
                                stddev=1.0 / math.sqrt(float(hiddensize2))),
            name='weights')
        biases = tf.Variable(tf.zeros([OUTPUT_SIZE]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases

    preds=tf.nn.softmax(logits)
    batch_xentropy =  tf.nn.softmax_cross_entropy_with_logits(labels=Y,
                                                  logits=logits,
                                                  name="xentropy")

    batch_loss=tf.reduce_mean(batch_xentropy)

    return w1, b1, w2, b2, logits, preds, batch_xentropy, batch_loss

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W,padding):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding=padding)


def convnet(X, Y, convlayer_sizes=[10, 10], \
        filter_shape=[3, 3], outputsize=10, hidden_size=100, padding="SAME"):
    """
    Create a Tensorflow model for a Convolutional Neural Network. The network
    should be of the following structure:
    conv_layer1 -> conv_layer2 -> fully-connected -> output

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        conv1: A convolutional layer of convlayer_sizes[0] filters of shape filter_shape
        conv2: A convolutional layer of convlayer_sizes[1] filters of shape filter_shape
        w: Connection weights for final layer
        b: biases for final layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch

    hints:
    1) consider tf.layer.conv2d
    2) the final layer is very similar to the onelayer network. Only the input
    will be from the conv2 layer. If you reshape the conv2 output using tf.reshape,
    you should be able to call onelayer() to get the final layer of your network
    """

    with tf.name_scope('reshape'):
        x_image = tf.reshape(X, [-1, 28, 28, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([filter_shape[0],filter_shape[1], 1, convlayer_sizes[0]])
        b_conv1 = bias_variable([convlayer_sizes[0]])
        conv1 = tf.nn.relu(conv2d(x_image, W_conv1,padding) + b_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([filter_shape[0],filter_shape[1], convlayer_sizes[0], convlayer_sizes[1]])
        b_conv2 = bias_variable([convlayer_sizes[1]])
        conv2 = tf.nn.relu(conv2d(conv1, W_conv2,padding) + b_conv2)

    conv_height = math.ceil(math.ceil(float(28) / 2.0)/2.0)
    conv_width = math.ceil(math.ceil(float(28) / 2.0) /2.0)#stride=2
    num_conf_features=conv_width * conv_height * convlayer_sizes[1]
    with tf.name_scope('layer1'):
        w = tf.Variable(tf.truncated_normal([num_conf_features, hidden_size],
                                stddev=1.0 / math.sqrt(float(hidden_size))), name='weights')
        b = tf.Variable(tf.zeros([hidden_size]),
                             name='biases')
        hidden = tf.nn.relu(tf.matmul(tf.reshape(conv2, [-1,num_conf_features]), w) + b)
        # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden_size, outputsize],
                                stddev=1.0 / math.sqrt(float(hidden_size))),
            name='weights')
        biases = tf.Variable(tf.zeros([outputsize]),
                             name='biases')
        logits = tf.matmul(hidden, weights) + biases

    preds=tf.nn.softmax(logits)
    batch_xentropy =  tf.nn.softmax_cross_entropy_with_logits(labels=Y,
                                                  logits=logits,
                                                  name="xentropy")

    batch_loss=tf.reduce_mean(batch_xentropy)

    return conv1, conv2, w, b, logits, preds, batch_xentropy, batch_loss

def train_step(sess, batch, X, Y, train_op, loss_op, summaries_op):
    """
    Run one step of training.

    :param sess: the current session
    :param batch: holds the inputs and target outputs for the current minibatch
    batch[0] - array of shape [minibatch_size, 784] with each row holding the
    input images
    batch[1] - array of shape [minibatch_size, 10] with each row holding the
    one-hot encoded targets
    :param X: the input placeholder
    :param Y: the output target placeholder
    :param train_op: the tensorflow operation that will run one step of training
    :param loss_op: the tensorflow operation that will return the loss of your
    model on the batch input/output

    :return: a 3-tuple: train_op_result, loss, summary
    which are the results of running the train_op, loss_op and summaries_op
    respectively.
    """
    train_result, loss, summary = \
        sess.run([train_op, loss_op, summaries_op], feed_dict={X: batch[0], Y: batch[1]})
    return train_result, loss, summary
