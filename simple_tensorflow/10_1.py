import tensorflow as tf
import numpy as np
from mido import MidiFile, MidiTrack
import sys
import os
sys.path.append(os.path.abspath('../with-mido')) # hop one directory over
from mido_final import vec2event, event2msg
sys.path.append(os.path.abspath('../simple-tensorflow')) # hop back

SIZE = 259

X = tf.placeholder(shape=(1, 10*SIZE,), dtype=tf.float32, name='Xin')
Y_ = tf.placeholder(shape=(1,SIZE,), dtype=tf.float32, name='Y_')

W = tf.Variable(tf.random_normal(shape=(10*SIZE,SIZE,))) # X*W has the same shape as B
b = tf.Variable(tf.random_normal(shape=(SIZE,))) # B has the same shape as Y and X

Y = tf.nn.softmax(X @ W + b)

cost = -tf.reduce_mean(Y_ * tf.log(Y))

init = tf.global_variables_initializer()

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

saver = tf.train.Saver((W,b,))
if sys.argv[1] == 'train':
    n=0
    for epoch in range(100):
        print('Epoch {} of 100'.format(n))
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, '10_1/-11') # latest session becuase there are 149 files
            for f in os.listdir('../processed/'):
                print('\tTraining on file {} in epoch {} of {}'.format(f, n, 100))
                vecs = np.load('../processed/'+f)
                y = vecs[10:]
                x = np.empty(shape=(len(vecs)-10, 10, SIZE))
                for i in range(len(vecs)-10):
                    x[i] = vecs[i:i+10]
                for i in range(len(vecs) - 11):
                    sess.run(train_step,feed_dict={X:x[i].reshape(1,-1), Y_:y[i].reshape(1,259)})
                saver.save(sess, '10_1/', global_step=n)
            n+=1

if sys.argv[1] == 'test':
    test_out = np.empty((100, SIZE,))
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, '10_1/-11')
        for i in range(15):
            test_out[i] = np.random.rand(SIZE)
        mid = MidiFile()
        trk = MidiTrack()
        mid.tracks.append(trk)
        for i in range(10,100 - 11):
            test_out[i+1] = sess.run(Y,feed_dict={X:test_out[i-10:i].reshape(1,10*259)})
            trk.append(event2msg(vec2event(test_out[i+1])))
        mid.save('test_10.mid')
