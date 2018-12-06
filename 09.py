import numpy as np
from meta import config
import helper
import dataset
import model
import os
import tensorflow as tf
import time
import tqdm
from IPython import embed

# --------- env setting--------
tf.set_random_seed(1234)
helper.auto_select_gpu(mem_bound=400, gpus=[0, 1, 2])
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
graph = tf.get_default_graph()
sess = tf.Session(graph=graph, config=tf_config)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 200, """training batch size""")
flags.DEFINE_float("init_lr", 0.001, """init learning rate""")
flags.DEFINE_float("momentum", 0.9, """momentum""")
flags.DEFINE_float("weight_decay", 0.01, """init weight decay""")

flags.DEFINE_integer("n_round", 10, """total rounds to achieve the target""")
flags.DEFINE_float("target", 0.25, """remain ratio""")
# flags.DEFINE_integer("print_train_interval", 90, """print train interval""")
flags.DEFINE_integer("n_total_epoch", 1000, """total training epoch""")
flags.DEFINE_float("steps_per_epoch", 250, """number of mini-batches in each epoch""")
flags.DEFINE_string("logdir", 'log/default', """checkpoint save interval""")
flags.DEFINE_integer("num_classes", 31, """number of classes to be classified""")
# flags.DEFINE_integer("checkpoint_interval", 5, """checkpoint save interval """)
# ----------- env setting done -------------------------

# get logger
helper.ensure_dir(FLAGS.logdir)
logger = helper.get_logger(FLAGS.logdir, 'train_log')

#  check parameters
FLAGS.logdir = os.path.abspath(os.path.expanduser(FLAGS.logdir))
logger.info('----------------- FLAGS ------------------------')
FLAGS._parse_flags()
kv_dict = {}
for k, v in FLAGS.__dict__['__flags'].items():
    logger.info('{}: {}'.format(k, v))
    kv_dict[k] = v
logger.info('------------------------------------------------')

config.num_output_classes = FLAGS.num_classes

# load dataset
ds_train = dataset.get_dataset('train', FLAGS.batch_size, True)
ds_test = dataset.get_dataset('test', None, False)

# inputs
image = tf.placeholder(name='images', dtype=tf.float32, shape=[None, config.features_dim])
label = tf.placeholder(name='label', dtype=tf.int64, shape=[None, config.num_output_classes])
lr = tf.placeholder(name='lr', shape=[], dtype=tf.float32)
weight_decay = tf.placeholder(name='weight_decay', shape=[], dtype=tf.float32)
mask_fc1 = tf.placeholder(name='mask_fc1', dtype=tf.float32, shape=[config.features_dim, config.hidden_dim])
mask_out = tf.placeholder(name='mask_out', dtype=tf.float32, shape=[config.hidden_dim, config.num_output_classes])
mask_fc1_data = np.ones(mask_fc1.get_shape().as_list())
mask_out_data = np.ones(mask_out.get_shape().as_list())
# build train model
logits = model.get_train_model(image, [mask_fc1, mask_out])
masked_weights = tf.get_collection('masked_weights')
losses = []
for v in masked_weights:
    losses.append(weight_decay * tf.reduce_sum(v ** 2))
loss_xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
losses.append(loss_xent)
loss = tf.add_n(losses)
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(label, axis=1)), tf.float32))

# train op
train_op = tf.train.AdamOptimizer(lr).minimize(loss)
# train_op = tf.train.MomentumOptimizer(lr, FLAGS.momentum).minimize(loss)

# summary
summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph)
tf.summary.histogram('fc1', masked_weights[0])
tf.summary.histogram('out', masked_weights[1])
summary_op = tf.summary.merge_all()


# begin to train
lr_data = FLAGS.init_lr
weight_decay_data = FLAGS.weight_decay
n_total_steps = FLAGS.steps_per_epoch * FLAGS.n_total_epoch
prune_ratio = 1 - FLAGS.target ** (1./(FLAGS.n_round-1))
new_prune_ratio = (1-FLAGS.target) / (FLAGS.n_round-1)
logger.info("Target: {}".format(FLAGS.target))
logger.info("Prune ratio for each round: {}".format(prune_ratio))
logger.info("New prune ratio for each round: {}".format(new_prune_ratio))

begin, timer = time.time(), time.time()
try:
    sess.run(tf.global_variables_initializer())
    params = tf.get_collection('params')
    for r in range(FLAGS.n_round):
        best_acc_data = 0.0
        best_params_data = []
        acc_list = []
        for i in tqdm.tqdm(range(n_total_steps)):
        #for i in range(n_total_steps):
            image_data, label_data = next(ds_train)
            helper.check_equal(len(image_data), FLAGS.batch_size)
            helper.check_equal(len(label_data), FLAGS.batch_size)
            data_time = time.time() - timer

            feed_dict_train = {
                image: image_data,
                label: label_data,
                lr: lr_data,
                mask_fc1: mask_fc1_data,
                mask_out: mask_out_data,
                weight_decay: weight_decay_data
            }

            # train
            _, loss_data, acc_data = sess.run([train_op, loss, acc], feed_dict=feed_dict_train)

            # print log
            # if i % FLAGS.steps_per_epoch == 0:
            #     logger.info("[{}.{}]loss: {:.5f}, acc: {:.5f} time: {:.3f}/{:.3f}, lr: {:.5f}".format(
            #         i // FLAGS.steps_per_epoch,
            #         i % FLAGS.steps_per_epoch,
            #         loss_data,
            #         acc_data,
            #         data_time,
            #         time.time() - timer - data_time,
            #         lr_data,
            #         ))
            # test
            if i % FLAGS.steps_per_epoch == 0:
                image_data, label_data = next(ds_test)
                feed_dict_test = {
                    image: image_data,
                    label: label_data,
                    mask_fc1: mask_fc1_data,
                    mask_out: mask_out_data
                }
                acc_data = sess.run(acc, feed_dict=feed_dict_test)
                if acc_data > best_acc_data:
                    best_acc_data = acc_data
                    best_params_data = sess.run(params)

                    # logger.info("Test, loss: {:.5f}, acc: {:.5f}".format(
                    #     np.mean(loss_data), np.mean(acc_data)
                    # ))
            timer = time.time()
        logger.info("Round {}, Best Accuracy: {}, weight_decay: {:.5g}, lr: {}".format(
            r, best_acc_data, weight_decay_data, lr_data))

        # restore best params first
        for i, p in enumerate(params):
            sess.run(tf.assign(p, best_params_data[i]))

        # tensorboard
        summary_data = sess.run(summary_op, feed_dict={
            mask_fc1: mask_fc1_data,
            mask_out: mask_out_data
        })
        summary_writer.add_summary(summary_data, global_step=r)

        # compute current model sparsity
        current_params_data = sess.run(params)
        logger.info('current remain: {:.5f}, {:.5f}'.format(np.mean(mask_fc1_data), np.mean(mask_out_data)))
        graph_tmp = model.save_constant_model(
            params=current_params_data,
            mask=[mask_fc1_data, mask_out_data],
            logdir=FLAGS.logdir,
            name=str(r),
            logger=logger
        )

        # update mask
        tmp1 = mask_fc1_data * current_params_data[0]
        tmp2 = mask_out_data * current_params_data[2]

        # prune by all
        #tmp1 = np.sort(np.abs(tmp1[tmp1 != 0]))
        #thr1 = tmp1[int(len(tmp1) * prune_ratio)]
        #tmp2 = np.sort(np.abs(tmp2[tmp2 != 0]))
        #thr2 = tmp2[int(len(tmp2) * prune_ratio)]

        # prune by cols
        #tmp1 = np.sort(np.abs(tmp1), axis=0)
        #thr1 = tmp1[int(tmp1.shape[0] * new_prune_ratio * (r+1)), :].reshape([1, -1])
        #tmp2 = np.sort(np.abs(tmp2), axis=0)
        #thr2 = tmp2[int(tmp2.shape[0] * new_prune_ratio * (r+1)), :].reshape([1, -1])
        #mask_fc1_data = mask_fc1_data * (np.abs(current_params_data[0]) > thr1).astype('float32')
        #mask_out_data = mask_out_data * (np.abs(current_params_data[2]) > thr2).astype('float32')

        # split cols for prune
        divided = tmp1.shape[0] // 2
        tmp1_1 = np.sort(np.abs(tmp1[0:divided, :]), axis=0)
        thr1_1 = tmp1_1[int(tmp1_1.shape[0] * new_prune_ratio * (r+1)), :].reshape([1, -1])
        tmp1_2 = np.sort(np.abs(tmp1[divided:, :]), axis=0)
        thr1_2 = tmp1_2[int(tmp1_2.shape[0] * new_prune_ratio * (r+1)), :].reshape([1, -1])
        mask_fc1_data = np.vstack([mask_fc1_data[0:divided, :] * (np.abs(current_params_data[0][0:divided, :]) > thr1_1).astype('float32'), mask_fc1_data[divided:, :] * (np.abs(current_params_data[0][divided:,:]) > thr1_2).astype('float32')])  

        tmp2 = np.sort(np.abs(tmp2), axis=0)
        thr2 = tmp2[int(tmp2.shape[0] * new_prune_ratio * (r+1)), :].reshape([1, -1])
        mask_out_data = mask_out_data * (np.abs(current_params_data[2]) > thr2).astype('float32')

except Exception as e:
    import logging
    logging.exception("message")
    logger.info(e)
finally:
    cost = time.time() - begin
    logger.info("Time elapse: {:d} hrs {:d} min {:0.2f} s".format(int(cost//3600), int((cost%3600)//60), cost%60))
    summary_writer.close()

