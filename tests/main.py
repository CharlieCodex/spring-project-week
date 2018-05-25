import tensorflow as tf
from utils import RNN

# one byte input, 256 possiblities for 1 hot encoding
SIZE_IN = 8
SEQ_LEN = 10
BATCH_SIZE = 10

cell = RNN(128, (SIZE_IN, 8,), (SIZE_IN,8,))

out_state, logits = 

Y_ = tf.placeholder(tf.float32, shape=(None,SIZE_IN,8,), name="Y_")

cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_)
train_step = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    sess.run((train_step, out_state), feed_dict="Stuff")
