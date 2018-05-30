import tensorflow as tf
from mido import MidiFile, MidiTrack
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import os
import sys
import numpy as np
import time
from mido_midware import mido_utils
import utils

tf.set_random_seed(100) # make things replicable
run_timestamp = int(time.time())

SEQLEN = 30
BATCHSIZE = 200
INTERNALSIZE = 256
# Vector encoding scheme 
# [dt (f32),
#  tempo(f32), 2
#  mode (f32, 3), 5
#  key(f32, 128), 133
#  vel(f32, 128)] 261
VECLEN = 261
NLAYERS = 10
learning_rate = 0.005

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

# This just sets up a simple set of weights and biases
# [ BATCHSIZE x SEQLEN, VECLEN ]
Ylogits = layers.linear(Yflat, VECLEN)

# split into:
# Io: int componenets (dt and tempo) [BATCH X SEQLEN, 2]
# Mo: mode componenet [BATCH X SEQLEN, 3]
# Ko: key componenet [BATCH X SEQLEN, 128]
# Vo: velocity componenet [BATCH X SEQLEN, 128]
Io, Mo, Ko, Vo = tf.split(Ylogits, [2, 3, 128, 128], 1)

softmaxed = tf.concat((
    tf.nn.softmax(Mo),
    tf.nn.softmax(Ko),
    tf.nn.softmax(Vo)), axis=1)

relued = tf.nn.relu(Io)

Yo = tf.concat((
    relued,
    softmaxed), axis=1, name='Yo')

# [ BATCHSIZE x SEQLEN, VECLEN ]
Yflat_ = tf.reshape(Yo_, [-1, VECLEN])    

ints, categories = tf.split(Yflat_, [2, 259], 1)

# [ BATCHSIZE x SEQLEN ]
loss1 = -tf.reduce_sum(categories*tf.log(softmaxed), axis=1)
nan_check = tf.is_nan(loss1)

loss2 = tf.reduce_sum((ints-relued)**2,axis=1)

loss = loss1 + loss2

# [ BATCHSIZE, SEQLEN ]
loss = tf.reshape(loss, [batchsize, -1])           

# [ BATCHSIZE x SEQLEN ]
Y = tf.argmax(Yo, 1)                          

# [ BATCHSIZE, SEQLEN ]
Y = tf.reshape(Y, [batchsize, -1], name="Y")  

# Adam is some crazy statstics optimizer that works
# sligtly better than gradient descent
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

SAVE_FREQ = 20
_50_BATCHES = SAVE_FREQ * BATCHSIZE * SEQLEN
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1000)


def gen_sample_file(n, name):
    print('\n\tGenerating sample file')
    ry = np.array([[utils.track_seed()]])
    rh = np.zeros([1, INTERNALSIZE * NLAYERS])
    mid = MidiFile()
    trk = MidiTrack()
    mid.tracks.append(trk)
    for k in range(n):
        # print(mido_utils.vec2msg(ry[0][0]))
        # generate a new state and output vec
        ryo, rh = sess.run([Yo, H], feed_dict={Xo: ry, Hin: rh, batchsize: 1})
        # sample the output vec into an argmaxed version
        rc = utils.prep_vec(ryo)
        # append this to our midi file as a mido.Message
        trk.append(mido_utils.vec2msg(rc))
        ry = np.array([[rc]])
    mid.save('samples/{}_{}.mid'.format(run_timestamp,name))

# init everything
# initial zero input state
istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
step = 0
nb_epochs = 50
for x, y_, epoch in utils.rnn_minibatch_sequencer(data, BATCHSIZE, SEQLEN, nb_epochs=nb_epochs):
    # train on one minibatch
    feed_dict = {Xo: x, Yo_: y_, Hin: istate, lr: learning_rate, batchsize: BATCHSIZE}
    _, y, ostate, nans, = sess.run([train_step, Y, H, nan_check], feed_dict=feed_dict)
    print('\rEpoch {:04d}/{:04d}, step {}\tnext save: {:04d}\tnext gen: {:04d}\tNaN check: {}'.format(
            epoch+1,
            nb_epochs,
            step,
            (10* _50_BATCHES - step % (10* _50_BATCHES)) // (BATCHSIZE * SEQLEN),
            (3 * _50_BATCHES - step % (3 * _50_BATCHES)) // (BATCHSIZE * SEQLEN),
            (True in nans)),
        end='')
    # print('\tBatch loss: {}'.format(np.mean(l)))
    # display a short text generated with the current weights and biases (every 150 batches)
    if step // 1 % _50_BATCHES == 0:
        gen_sample_file(500, step)

    # save a checkpoint (every 500 batches)
    if step // 10 % _50_BATCHES == 0:
        saved_file = saver.save(sess, 'checkpoints/rnn_train_{}'.format(run_timestamp), global_step=step)
        print("\n\tSaved file: " + saved_file)

    # loop state around h_out -> h_in
    istate = ostate
    step += BATCHSIZE * SEQLEN

saved_file = saver.save(sess, 'checkpoints/rnn_train_{}_final'.format(run_timestamp))
print("Final save: " + saved_file)
gen_sample_file(5000, 'final')
