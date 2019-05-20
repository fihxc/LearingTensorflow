import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import tensorflow as tf

# lstm_size可以认为是hidden层神经元数量
lstm_size=20

batch_size=50
test_size=100

def model(x,W,B,lstm_size):
    # 建立LSTM记忆层，即hidden层
    lstm=tf.contrib.rnn.BasicLSTMCell(lstm_size)

    # 将输入的第一维与第二维交换，是由tf.contrib.rnn.static_rnn函数的计算方式决定的，其输入形状为（time_size,batch_size,input_size）,
    # 在本次计算中，即（28,50,28），在计算时，每张图为28×28的序列，该函数第一步分别将50张图的第一个元素输入网络，得到50张图第一次计算的
    # hidden层（大小为50×lstm_size），结合该层和50张图的第二个元素进行第二次计算，得到第二次计算的hidden层，以此类推，最终得到（28,50,lstm_size）
    # 的结果，即每一次计算的hidden层
    xt=tf.transpose(x,[1,0,2])

    xr=tf.reshape(xt,[-1,28])

    x_split=tf.split(xr,28,axis=0)

    outputs,_states = tf.contrib.rnn.static_rnn(lstm,x_split,dtype=tf.float32)
    
    # 以上output只是所有图所有计算得到的hidden层合集，要得到最终的计算结果，需要最后一次计算得到的hidden层（即outputs[-1]）结合权重和偏置计算
    return tf.matmul(outputs[-1],W) + B,lstm.state_size

# 得到训练和测试数据
mnist=input_data.read_data_sets("./MNIST_data/",one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# tensorflow官方教程中输入大小是（-1,784），每张图片的784像素被一次输入网络进行前向传播得到结果;输入LSTM网络时将每张图的像素切割为28×28的序列，
# 进行28次前向传播，每次输入28个元素，最终得到结果
trX = trX.reshape(-1, 28, 28)
teX = teX.reshape(-1, 28, 28)

X = tf.placeholder("float", [None, 28, 28])
Y = tf.placeholder("int64", [None, 10])

# LSTM最后一个元素输出的hidden层需要进行向量和偏置的计算得到真正的输出，由于每个元素输出hidden层包含lstm_size个神经元，
# 因此W的第一个维度为lstm_size
W=tf.Variable(tf.random_normal([lstm_size,10]))
B=tf.Variable(tf.random_normal([10]))

py_x,state_size=model(X,W,B,lstm_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

predict_op = tf.argmax(py_x, 1)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(py_x, 1), Y)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        s=len(teX)
        test_indices = np.arange(len(teX))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices]})))
--------------------- 
作者：m米咔00 
来源：CSDN 
原文：https://blog.csdn.net/liu3612162/article/details/83997967 
版权声明：本文为博主原创文章，转载请附上博文链接！