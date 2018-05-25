from mido import MidiFile, Message
import numpy as np
from os import listdir
import reduce

def parse_events(f):
    midi = MidiFile(f)
    events = []
    for msg in midi:
        if msg.type == 'note_on' or msg.type == 'note_off':
            events.append({
                'type': msg.type,
                'dt': msg.time,
                'key': msg.note,
                'vel': msg.velocity})
    return events

def event2vec(event):
    '''Transform a single event into a vector of length 290'''
    dt = np.array([int(x) for x in '{:032b}'.format(int(event['dt']))]) # list of 0 and 1's
    on_off = np.array((
        0 if event['type'] == 'note_off' else 1,
        1 if event['type'] == 'note_off' else 0,)) # onehot encoding for on/off
    key = np.zeros((128,))
    key[event['key']] = 1
    vel = np.zeros((128,))
    key[event['vel']] = 1
    return np.concatenate((dt, on_off, key, vel))

def vec2event(vec):
    '''Transform a vector of length 290 into a midi event'''
    vec_dt = vec[:32] # get first 32 components [binary]
    vec_on_off = vec[32:34] # get on/of components [1 hot]
    vec_key = vec[34:34+128] # get key components [1 hot]
    vec_vel = vec[34+128:] # get velocity components [1 hot]
    str_dt = [str(e) for e in vec_dt] # convert first 32 components into strings
    dt = int(''.join(str_dt),2) # join strings and cast as base 2 integer
    on_off = 'note_off' if vec_on_off[0] else 'note_on' # extract on/off -ness of note
    key = np.argmax(vec_key) # get the 1-hot index
    vel = np.argmax(vec_vel) # get the 1-hot index
    return {'type': on_off, 'dt': dt, 'key': key, 'vel': vel}

def event2msg(event):
    '''Convert an internally used event dict into a mido midi message'''
    return Message(event.type,
        time=event['dt'],
        note=event['key'],
        velocity=event['vel'])

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
