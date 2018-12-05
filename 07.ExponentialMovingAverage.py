import tensorflow as tf

MOVING_AVERAGE_DECAY = 0.99

global_step = tf.Variable(0, trainable=False)

w = tf.Variable(tf.constant(5, dtype=tf.float32))

ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

ema_op = ema.apply(tf.trainable_variables())

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    
    sess.run(init_op)
    print(sess.run([w, ema.average(w)]))

    sess.run(tf.assign(w, 1))
    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))
    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))

    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w, 10))
    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))
    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))
    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))
    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))
    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))
    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))
    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))