import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

lstm_cell = tf.nn.rnn_cell.LSTMCell(
    num_units = 128,
    
)