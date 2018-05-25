import numpy as np
from counted_buffer import CountedBuffer
from utils import read_header, read_chunk
from os import listdir

def parse_events(f):
    '''Parse entire midifile into events'''
    buf = CountedBuffer(open(f,'rb'))
    header = read_header(buf)
    events = []
    print(buf.peek(4))
    while buf.peek(4) == b'MTrk':
        events += read_chunk(buf)
    return header, events

def event2vec(event):
    '''Transform a single event into a vector of length 290'''
    dt = np.array([int(x) for x in '{:032b}'.format(event['dt'])]) # list of 0 and 1's
    on_off = np.array((
        0 if event['type'] == 'note_off' else 1,
        1 if event['type'] == 'note_off' else 0,)) # onehot encoding for on/off
    key = np.zeros((128,))
    key[event['key']] = 1
    vel = np.zeros((128,))
    key[event['vel']] = 1
    return np.concatenate((dt, on_off, key, vel))

def translate_file(in_path, out_path):
    '''Read midi file from in_path and create an np array file at out_path'''
    _, events = parse_events(in_path)
    vecs = np.array([event2vec(e) for e in events])
    np.save(out_path, vecs)

def translate_dir(in_dir, out_dir):
    '''Read all midi files in in_dir and create corrosponding .npy vec files in out_dir'''
    for in_path in listdir(in_dir):
        print('Processing {}'.format(in_path))
        translate_file(in_dir+in_path, out_dir+in_path.rstrip('.mid')+'.npy')


if __name__ == '__main__':
    h, e = parse_events('../data/alb_esp1.mid')