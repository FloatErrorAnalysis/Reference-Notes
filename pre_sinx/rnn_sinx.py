import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

training_examples = 10000
test_examples = 1000
# gap of sinx
sample_gap = 0.01
time_steps = 20


# Generate the data of sin(x)
# For the input sequence of [X1, X2..., Xtime_steps]
# Given the result of Y
# The result sequence[X1, X2... Xtime_steps, Y]
def generate_data(seq):
    X = []
    y = []
    for i in range(len(seq) - 1 - time_steps):
        X.append(seq[i: i + time_steps])
        y.append(seq[i + time_steps])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# Train(0, start)
# Test(start, end)
# number: training_examples * sample_gap

test_start = training_examples * sample_gap
test_end = training_examples * sample_gap + test_start
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, training_examples)))
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, training_examples)))


# Set the size, output
lstm_size = 30
lstm_layers = 2
bath_size = 64

x = tf.placeholder(tf.float32, [None, time_steps, 1], name='input_x')
y = tf.placeholder(tf.float32, [None, 1], name='input_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(lstm_size)


lstm = lstm_cell()
drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
# Add the layers
cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(0, lstm_layers)])


outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype=np.float32)
outputs = outputs[:, -1]
predictions = tf.contrib.layers.fully_connected(outputs, 1, activation_fn=tf.tanh)
cost = tf.losses.mean_squared_error(y, predictions)
optimizer = tf.train.AdamOptimizer().minimize(cost)


def get_baches(X, y, batch_size = 64):
    for i in range(0, len(X), batch_size):
        begin_i = i
        end_i = i + batch_size if i + batch_size < len(X) else len(X)

    yield X[begin_i: end_i], y[begin_i: end_i]


# Train
echos = 20
session = tf.Session()
iteration = 1
with session.as_default() as ss:
    tf.global_variables_initializer().run()

    for e in range(0, echos):
        for xs, ys in get_baches(train_X, train_y, bath_size):
            feed_dict = {x: xs[:, :, None], y: ys[:, None], keep_prob:.5}

            loss, _ = ss.run([cost, optimizer], feed_dict=feed_dict)
            if iteration % 100 == 0:
                print('Epochs:{}/{}'.format(e, echos),
                      'Iteration:{}'.format(iteration),
                      'Train loss: {:.8f}'.format(loss))
            iteration += 1
