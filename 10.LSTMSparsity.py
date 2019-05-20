import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import input_data

input_vec_size = 28
lstm_size = 28
time_step_size = 28

batch_size = 128
test_size = 256
prune_ratio = 0.6

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, W, B, lstm_size):
    XT = tf.transpose(X, [1, 0, 2])
    XR = tf.reshape(XT, [-1, lstm_size])
    X_split = tf.split(XR, time_step_size, 0)
    lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0,state_is_tuple=True)
    # A = lstm.get_kernel()
    # print("A ", A)
    print(lstm)
    outputs, _states = rnn.static_rnn(lstm, X_split, dtype=tf.float32)
    print(lstm)
    A = lstm.get_kernel()
    print("A ", A)
    return tf.matmul(outputs[-1],W)+B, lstm.state_size, lstm

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

trX = trX.reshape(-1, 28, 28)
teX = teX.reshape(-1, 28, 28)

X = tf.placeholder("float", [None, 28, 28])
Y = tf.placeholder("float", [None, 10])

W = tf.Variable(init_weights([lstm_size, 10]))
B = tf.Variable(init_weights([10]))

py_x, state_size, lstm = model(X, W, B, lstm_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

predict_op = tf.argmax(py_x, 1)

def prune(value ,prune_ratio):
    tmp = np.sort(np.abs(value), None)
    thr = tmp[int(len(tmp)*prune_ratio)]
    mask = np.abs(value) > thr
    return mask*value
 
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1,batch_size)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start: end]})

        #prune W
        W_value = sess.run(W)
        W_new = prune(W_value, prune_ratio)
        sess.run(tf.assign(W, W_new))

        #prune kernel
        kernel = sess.run(lstm.weights[0])
        kernel = prune(kernel, prune_ratio)
        sess.run(tf.assign(lstm.weights[0], kernel))

        #prune bias
        bias = sess.run(lstm.weights[1])
        bias = prune(bias, prune_ratio)
        sess.run(tf.assign(lstm.weights[1], bias))

        s = len(teX)
        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0: test_size]
        print(i, np.mean(np.argmax(teY[test_indices], axis=1) == sess.run(predict_op, feed_dict={X: teX[test_indices]})))
