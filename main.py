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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from plot_utils import AutoPlotAxes


with tf.variable_scope('g') as scope:
    from model import *

tf.set_random_seed(1023)
run_timestamp = int(time.time())

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")

int_accr

saver = tf.train.Saver(max_to_keep=1000)

save_path = tf.train.latest_checkpoint('checkpoints')
print('Latest save, {}'.format(save_path))

if len(sys.argv) == 2:
    if not sys.argv[1] == '--no-load':
        print('Loading from specified checkpoint {}'.format(sys.argv[1]))
        saver.restore(sess, sys.argv[1])
    else:
        print('Starting blank due to option {}'.format(sys.argv[1]))
elif save_path:
    print('Loading from latest checkpoint {}'.format(save_path))
    saver.restore(sess, save_path)

data_dir = "chpn_monokey/chpn_op10_e01.npy"
data, trackranges = utils.read_data_files(data_dir)

epoch_size = len(data) // (BATCHSIZE * SEQLEN)
SAVE_FREQ = 1
_50_BATCHES = SAVE_FREQ * BATCHSIZE * SEQLEN

def gen_sample_file(n, name):
    print('\n\tGenerating sample file')
    ry = np.array([[utils.track_seed()]])
    rh = np.zeros([1, INTERNALSIZE * NLAYERS])
    mid = MidiFile()
    trk = MidiTrack()
    mid.tracks.append(trk)
    flubbs = 0
    cur_notes = 0
    non_zero = 0
    for _ in range(n):
        # print(mido_utils.vec2msg(ry[0][0]))
        # generate a new state and output vec
        ryo, rh = sess.run([Yo, H], feed_dict={Xo: ry, Hin: rh, batchsize: 1, pkeep: 1})
        if True in np.isnan(ryo[0]):
            continue
        # sample the output vec into an argmaxed version
        rc = utils.prep_vec(ryo)
        if rc[0] > 0:
            non_zero += 1
        if rc[0]==0:
            cur_notes += 1
            if cur_notes > 20:
                rc = utils.random_note(dt=256)
                cur_notes = 0
                flubbs+=1
        # append this to our midi file as a mido.Message
        trk.append(mido_utils.vec2msg(rc))
        ry = np.array([[rc]])
    mid.save('samples/{}_{}.mid'.format(run_timestamp, name))
    print('flubbs/n {:04.4f}%, nonzero: {:04.4f}%'.format(100*flubbs/n, 100*non_zero/n))
# init everything
# initial zero input state
nb_epochs = 100

class Anim:
    def __init__(self, sess, saver, loss_ax: plt.Axes, accuracy_ax: plt.Axes):
        self.loss = None
        self.category = None
        self.ints  = None
        self.acc  = None
        self.step = 0
        self.accuracy_axes = accuracy_ax
        self.accuracy_axes.set_xbound(lower=0, upper=10)
        self.loss_axes = loss_ax
        self.loss_axes.set_xbound(lower=0, upper=10)
        self.istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])
        self.sess = sess
        self.saver = saver

    def do_train_step(self, tpl):
        x, y_, epoch = tpl
        # train on one minibatch
        feed_dict = {Xo: x, Yo_: y_,
                     Hin: self.istate, lr: learning_rate,
                     batchsize: BATCHSIZE, pkeep: 1.0}
        fetch_list= [train_step, H, nan_check,
                     loss, loss1, loss2,
                     accuracy, int_accr, mode_accr, key_accr, vel_accr]
        _, ostate, nans, l, l1, l2, acc, ia, ma, ka, va = self.sess.run(fetch_list, feed_dict=feed_dict)
        print('\rEpoch {:04d}/{:04d}, step {}\tnext save: {:04d}\tnext gen: {:04d}\tNaN check: {}'.format(
                epoch+1,
                nb_epochs,
                self.step,
                (15 - self.step // _50_BATCHES % 10) % 10,
                (10 - self.step // _50_BATCHES % 10) % 10,
                (True in nans)),
            end='')
        l = np.mean(l)
        l1 = np.mean(l1)
        l2 = np.mean(l2)
        acc = np.mean(acc)
        
        assert np.all(np.isfinite(l1)), 'Loss1 nan'
        assert np.all(np.isfinite(l2)), 'Loss2 nan'
        assert np.all(np.isfinite(l)), 'Loss nan'
        assert np.all(np.isfinite(ia)), 'Iacc nan'
        assert np.all(np.isfinite(ma)), 'Macc nan'
        assert np.all(np.isfinite(ka)), 'kacc nan'
        assert np.all(np.isfinite(va)), 'Vacc nan'
        assert np.all(np.isfinite(acc)), 'Acc nan'

        small_step = self.step // (BATCHSIZE * SEQLEN)
        
        self.loss.update(small_step, l)
        self.category.update(small_step, l1)
        self.ints.update (small_step, l2)
        
        self.acc.update(small_step, acc)
        self.acc_i.update(small_step, np.mean(ia))
        self.acc_m.update(small_step, np.mean(ma))
        self.acc_k.update(small_step, np.mean(ka))
        self.acc_v.update(small_step, np.mean(va))
        # print('\tBatch loss: {}'.format(np.mean(l)))
        # display a short text generated with the current weights and biases (every 150 batches)
        if self.step // _50_BATCHES %  10 == 0:
            if self.step // _50_BATCHES % 50 == 0:
                gen_sample_file(5000, '{}_long'.format(self.step))
            else:
                gen_sample_file(300, self.step)

        # save a checkpoint (every 500 batches)
        if self.step // _50_BATCHES % 10 == 5:
            saved_file = self.saver.save(self.sess,
                'checkpoints/rnn_train_{}'.format(run_timestamp),
                global_step=self.step)
            print("\n\tSaved file: " + saved_file)

        # loop state around h_out -> h_in
        self.istate = ostate
        self.step += BATCHSIZE * SEQLEN
        return (self.category._line, self.ints._line,
            self.acc_i, self.acc_m, self.acc_k, self.acc_v,
            self.loss._line, self.acc._line,)
    def init_func(self):
        self.loss = AutoPlotAxes(self.loss_axes, 'k-',
            label='Total loss')
        self.category = AutoPlotAxes(self.loss_axes.twinx(), 'r-',
            label='Category loss')
        self.ints  = AutoPlotAxes(self.loss_axes.twinx(), 'g-',
            label='Int loss')
        self.accuracy_axes.set_ylim(0, 1)
        self.acc  = AutoPlotAxes(self.accuracy_axes, 'k-', False,
            label='Average Accuracy')
        self.acc_i  = AutoPlotAxes(self.accuracy_axes, 'y-', False,
            label='Int Accuracy')
        self.acc_v  = AutoPlotAxes(self.accuracy_axes, 'b-', False,
            label='Velocity Accuracy')
        self.acc_m  = AutoPlotAxes(self.accuracy_axes, 'r-', False,
            label='Mode Accuracy')
        self.acc_k  = AutoPlotAxes(self.accuracy_axes, 'g-', False,
            label='Key Accuracy')
        return (self.loss._line, self.category._line, self.ints._line, self.acc._line,
            self.acc_i, self.acc_m, self.acc_k, self.acc_v,)

fig = plt.figure()
fig.canvas.set_window_title('Tensorflow graphs')
ax1 = plt.subplot2grid((2, 1), (0, 0))
ax1.set_title('Loss vs step')
ax2 = plt.subplot2grid((2, 1), (1, 0))
ax2.set_title('Accuracy vs step')
a = Anim(sess, saver, ax1, ax2)
anim = FuncAnimation(fig, a.do_train_step,
    frames=utils.rnn_minibatch_sequencer(data, BATCHSIZE, SEQLEN, nb_epochs=nb_epochs),
    init_func=a.init_func,
    interval=100)

def save_fig(evt):
    fig.savefig('charts/loss_accuracy_{}_final.png'.format(run_timestamp))

fig.canvas.mpl_connect('close_event', save_fig)
plt.show()
print('\nLeaving animation loop')
gen_sample_file(3000, 'final')
saved_file = saver.save(sess, 'checkpoints/rnn_train_{}_final'.format(run_timestamp))
print("\nFinal save: " + saved_file)