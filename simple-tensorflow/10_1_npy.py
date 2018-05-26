import numpy as np
from mido import MidiFile, MidiTrack
import sys
import os
sys.path.append(os.path.abspath('../with-mido')) # hop one directory over
from mido_final import vec2event, event2msg
sys.path.append(os.path.abspath('../simple-tensorflow')) # hop back

SIZE = 259

X = np.empty(shape=(1, 10*SIZE,), dtype=np.float32)
Y_ = np.empty(shape=(1,SIZE,), dtype=np.float32)

W = np.random.rand(10*SIZE,SIZE) # X*W has the same shape as B
b = np.random.rand(1,SIZE) # B has the same shape as Y and X

# softmax activiation function
def softmax(vec):
    return np.exp(vec)/np.sum(np.exp(vec))

# softmax derivative, either for specific indicies or as a matrix
# this is also dYdS
def softmax_der(vec, i=None, j=None):
    if i == None and j == None:
        out = np.empty((vec.size, vec.size,))
        for i  in range(vec.size):
            for j in range(vec.size):
                out[i][j] = softmax_der(vec, i, j)
        return out
    return np.exp(vec[0][i])*(np.exp(vec[0][j])-np.sum(np.exp(vec)))/ \
        (np.sum(np.exp(vec)) ** 2).reshape(1,1)

# derivative of 'S' with respect to W
def dSdW(Y,X,W,b):
    return X.T

# derivative of 'S' with respect to B
def dSdB(Y,X,W,b):
    return np.identity(b.size)

def dCdY(Y_,Y):
    return -Y_ / Y

learning_rate = 0.001
def train_step(rate, Y_, X, W, b):
    '''Find derivatives and do gradient descent'''
    Y = softmax(X @ W + b)
    # dC/dW = dS/dW . dC/dY . dY/dS
    # (nxn) = (nx1) . (1xn) . (nxn)
    # (nxn) =     (nxn)   .   (nxn)
    dCdW = dSdW(Y, X, W, b) @ dCdY(Y_, Y) @ softmax_der(Y)
    dCdB = dCdY(Y_, Y) @ softmax_der(Y)
    print('\t\tTraining step complete, cost: ', -np.sum(Y_*np.log(Y)))
    W -= rate*dCdW
    b -= rate*dCdB

if sys.argv[1] == 'train':
    '''Train on all available data'''
    n=0
    nb_epochs = 1
    for epoch in range(nb_epochs):
        print('Epoch {} of 1'.format(n))
        for f in os.listdir('../processed/'):
            print('\tTraining on file {} in epoch {} of {}'.format(f, n, nb_epochs))
            vecs = np.load('../processed/'+f)
            y = vecs[10:]
            x = np.empty(shape=(len(vecs)-10, 10, SIZE))
            for i in range(len(vecs)-10):
                x[i] = vecs[i:i+10]
            for i in range(len(vecs) - 11):
                train_step(learning_rate, y[i].reshape(1,259),x[i].reshape(1,-1), W, b)
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
