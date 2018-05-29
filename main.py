import tensorflow as tf
from mido import MidiFile, MidiTrack
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import os
import numpy as np
import time
from mido_midware import mido_utils
import utils

tf.set_random_seed(0) # make things replicable
run_timestamp = int(time.time())

SEQLEN = 30
BATCHSIZE = 200
INTERNALSIZE = 512
VECLEN = 259
NLAYERS = 3
learning_rate = 0.001

data_dir = "processed/*.npy"
# TODO WRITE MINIBATCH SEQUENCER
# mock code from martin gorner
data, trackranges = utils.read_data_files(data_dir)

epoch_size = len(data) // (BATCHSIZE * SEQLEN)

lr = tf.placeholder(tf.float32, name='lr')  # learning rate
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs
Xo = tf.placeholder(tf.float32, [None, None, VECLEN], name='X')    # [ BATCHSIZE, SEQLEN ]
# expected outputs = same sequence shifted by 1 since we are trying to predict the next character
Yo_ = tf.placeholder(tf.float32, [None, None, VECLEN], name='Y_')  # [ BATCHSIZE, SEQLEN ]
# input state
Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

# using a NLAYERS=3 layers of GRU cells, unrolled SEQLEN=30 times
# dynamic_rnn infers SEQLEN from the size of the inputs Xo

# How to properly apply dropout in RNNs: see README.md
cells = [rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]
# "naive dropout" implementation
multicell = rnn.MultiRNNCell(cells, state_is_tuple=False)

Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
# Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
# H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ]
# this is the last state in the sequence

H = tf.identity(H, name='H')  # just to give it a name

# [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])  

# [ BATCHSIZE x SEQLEN, VECLEN ]
Ylogits = layers.linear(Yflat, VECLEN)  

# [ BATCHSIZE x SEQLEN, VECLEN ]
Yflat_ = tf.reshape(Yo_, [-1, VECLEN])    

# [ BATCHSIZE x SEQLEN ]
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)

# [ BATCHSIZE, SEQLEN ]
loss = tf.reshape(loss, [batchsize, -1])      

# [ BATCHSIZE x SEQLEN, VECLEN ]
Yo = tf.nn.relu(Ylogits, name='Yo')        

# [ BATCHSIZE x SEQLEN ]
Y = tf.argmax(Yo, 1)                          

# [ BATCHSIZE, SEQLEN ]
Y = tf.reshape(Y, [batchsize, -1], name="Y")  

# Adam is some crazy statstics optimizer that works
# sligtly better than gradient descent
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

SAVE_FREQ = 50
_50_BATCHES = SAVE_FREQ * BATCHSIZE * SEQLEN
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1000)

# init everything
# initial zero input state
istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0
nb_epochs = 1000
for x, y_, epoch in utils.rnn_minibatch_sequencer(data, BATCHSIZE, SEQLEN, nb_epochs=nb_epochs):
    print('Epoch {} of {}, step {}'.format(epoch+1, nb_epochs, step))
    # train on one minibatch
    feed_dict = {Xo: x, Yo_: y_, Hin: istate, lr: learning_rate, batchsize: BATCHSIZE}
    _, y, ostate = sess.run([train_step, Y, H], feed_dict=feed_dict)

    # display a short text generated with the current weights and biases (every 150 batches)
    if step // 3 % _50_BATCHES == 0:
        print('\tGenerating sample file')
        ry = np.array([[utils.track_seed()]])
        rh = np.zeros([1, INTERNALSIZE * NLAYERS])
        mid = MidiFile()
        trk = MidiTrack()
        mid.tracks.append(trk)
        for k in range(1000):
            # generate a new state and output vec
            ryo, rh = sess.run([Yo, H], feed_dict={Xo: ry, Hin: rh, batchsize: 1})
            # sample the output vec into an argmaxed version
            rc = utils.prep_vec(ryo)
            # append this to our midi file as a mido.Message
            trk.append(mido_utils.vec2msg(rc))
            ry = np.array([[rc]])
        mid.save('samples/{}_{}.mid'.format(run_timestamp,step))

    # save a checkpoint (every 500 batches)
    if step // 10 % _50_BATCHES == 0:
        saved_file = saver.save(sess, 'checkpoints/rnn_train_{}'.format(run_timestamp), global_step=step)
        print("\tSaved file: " + saved_file)

    # loop state around h_out -> h_in
    istate = ostate
    step += BATCHSIZE * SEQLEN