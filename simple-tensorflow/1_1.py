import tensorflow as tf
import numpy as np
from mido import MidiFile, MidiTrack
import sys
import os
sys.path.append(os.path.abspath('../with-mido')) # hop one directory over
from mido_final import vec2event, event2msg
sys.path.append(os.path.abspath('../simple-tensorflow')) # hop back

SIZE = 259

X = tf.placeholder(shape=(1,SIZE,), dtype=tf.float32, name='Xin')
Y_ = tf.placeholder(shape=(1,SIZE,), dtype=tf.float32, name='Y_')

W = tf.Variable(tf.random_normal(shape=(SIZE,SIZE,))) # X*W has the same shape as B
b = tf.Variable(tf.random_normal(shape=(SIZE,))) # B has the same shape as Y and X

Y = tf.nn.softmax(X @ W + b)

cost = -tf.reduce_mean(Y_ * tf.log(Y))

init = tf.global_variables_initializer()

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

vecs = np.load('../processed/appass_2.npy')

x = vecs[:-1]
y = vecs[1:]

saver = tf.train.Saver((W,b,))

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(vecs) - 1):
        sess.run(train_step,feed_dict={X:x[i].reshape(1,259), Y_:y[i].reshape(1,259)})
    saver.save(sess, '1_1/', global_step=i)

test_out = np.empty_like(y)
with tf.Session() as sess:
    sess.run(init)
    test_out[0] = x[0]
    mid = MidiFile()
    trk = MidiTrack()
    mid.tracks.append(trk)
    for i in range(len(vecs) - 2):
        test_out[i+1] = sess.run(Y,feed_dict={X:test_out[i].reshape(1,259)})
        trk.append(event2msg(vec2event(test_out[i+1])))
    mid.save('test.mid')
