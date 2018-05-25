from mido import MidiFile
import numpy as np
from os import listdir

def parse_events(f):
    midi = MidiFile(f)
    events = []
    for e in midi:
        if e.type == 'note_on' or e.type == 'note_off':
            events.append({'type': e.type, 'dt': e.time, 'key':e.note, 'vel': e.velocity, 'dt': e.time})
    return events

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
    events = parse_events(in_path)
    vecs = np.array([event2vec(e) for e in events])
    np.save(out_path, vecs)

def translate_dir(in_dir, out_dir):
    '''Read all midi files in in_dir and create corrosponding .npy vec files in out_dir'''
    for in_path in listdir(in_dir):
        print('Processing {}'.format(in_path))
        translate_file(in_dir+in_path, out_dir+in_path.rstrip('.mid')+'.npy')
