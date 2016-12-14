import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

dataset = input_data.read_data_sets('/tmp/data/', one_hot = True)

class NetworkLayer():
    def __init__(self, n_nodes, weights, biases, activation = lambda x: x):
        self.n_nodes = n_nodes
        self.activation = activation
        self.weights = weights
        self.biases = biases

train_epochs = 3
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128
_, number_of_features = dataset.train.images.shape
_, number_of_classes = dataset.train.labels.shape

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

network = [NetworkLayer(n_nodes=rnn_size, weights=[], biases=[], activation=tf.nn.relu)]

def reccurent_neural_network(data):
    layer_output = data
    
    for layer in network:
        layer.weights = tf.Variable(tf.random_normal([rnn_size, number_of_classes]))
        layer.biases = tf.Variable(tf.random_normal([number_of_classes]))

        data = tf.transpose(data, [1,0,2])
        data = tf.reshape(x, [-1, chunk_size])
        data = tf.split(0, n_chunks, data)

        lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
        outputs, states = rnn.rnn(lstm_cell, data, dtype=tf.float32)

        layer_output = layer.activation(tf.matmul(outputs[-1], layer.weights) + layer.biases)

    return layer_output

def train_neural_network(x):
    prediction = reccurent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    # default learning rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(train_epochs):
            epoch_cost = 0

            for _ in range(int(dataset.train.num_examples/batch_size)):
                epoch_x, epoch_y = dataset.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_cost += c

            print('Epoch', epoch, 'completed out of', train_epochs, 'loss:', epoch_cost)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: dataset.test.images.reshape((-1, n_chunks, chunk_size)), y: dataset.test.labels}))

train_neural_network(x)