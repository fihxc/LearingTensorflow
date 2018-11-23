import tensorflow as tf

a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])

c = a + b

print(c)

x = tf.constant([[1.0, 2.0]])
w = tf.constant([[3.0], [4.0]])

xw = tf.matmul(x, w)

print(xw)

with tf.Session() as sess:
    res = sess.run(xw)
    print(res)