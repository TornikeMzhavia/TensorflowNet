import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Sentiment_data import create_feature_sets_and_labels

train_x, train_y, test_x, test_y = create_feature_sets_and_labels(os.path.join(os.path.dirname(__file__), 'pos.txt'), 
                                                                  os.path.join(os.path.dirname(__file__), 'neg.txt'))

class NetworkLayer():
    def __init__(self, n_nodes, weights, biases, activation = lambda x: x):
        self.n_nodes = n_nodes
        self.activation = activation
        self.weights = weights
        self.biases = biases

batch_size = 100
number_of_features = len(train_x[0])
number_of_classes = 2

x = tf.placeholder('float', [None, number_of_features])
y = tf.placeholder('float')

network = [NetworkLayer(n_nodes=500, weights=[], biases=[], activation=tf.nn.relu),
           NetworkLayer(n_nodes=300, weights=[], biases=[], activation=tf.nn.relu),
           NetworkLayer(n_nodes=200, weights=[], biases=[], activation=tf.nn.relu),
           NetworkLayer(n_nodes=number_of_classes, weights=[], biases=[])]

def neural_network_model(data):
    layer_output = data
    
    for layer in network:
        layer.weights = tf.Variable(tf.random_normal([int(layer_output.get_shape()[1]), layer.n_nodes]))
        layer.biases = tf.Variable(tf.random_normal([layer.n_nodes]))

        layer_output = layer.activation(tf.matmul(layer_output, layer.weights) + layer.biases)

    return layer_output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    # default learning rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    train_epochs = 15
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(train_epochs):
            epoch_cost = 0

            i=0
            while i < len(train_x):
                start = i
                end = i + batch_size

                epoch_x = np.array(train_x[start:end])
                epoch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_cost += c

                i += batch_size

            print('Epoch', epoch, 'completed out of', train_epochs, 'loss:', epoch_cost)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

train_neural_network(x)