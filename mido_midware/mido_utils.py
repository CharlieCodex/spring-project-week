from mido import MidiFile, MidiTrack, MetaMessage, Message, merge_tracks
import numpy as np
from glob import glob
from os.path import basename, exists
from os import mkdir
import sys

# keys to semitones away from A major
key_semitone_dict = {
    'A' : 0,
    'A#': 1,
    'Bb': 1,
    'B' : 2,
    'C' : 3,
    'C#': 4,
    'Db': 4,
    'D' : 5,
    'D#': 6,
    'Eb': 6,
    'E' : -5,
    'F' : -4,
    'F#': -3,
    'Gb': -3,
    'G' : -2,
    'G#': -1,
    'Ab': -1,
}

def key_to_semitone_offset(key: str):
    '''Convert from a mido key str into a semitone offset'''
    offset = 0
    if key.startswith('m'):
        offset += 3
        key = key[1:]
    return offset + key_semitone_dict[key]

def parse_events(f):
    midi = MidiFile(f)
    events = []
    tracks = [midi.tracks[0]]
    tracks.extend([track for track in midi.tracks if 'piano' in track.name.lower()])
    key_offset = 0
    key_set = False
    for msg in merge_tracks(tracks):
        if key_set is False:
            if msg.type == 'key_signature':
                key_offset = key_to_semitone_offset(msg.key)
                key_set = True
            else:
                events.append(msg2event(msg, key_offset))
    return events

def msg2event(msg, key_offset):
    if msg.type == 'set_tempo':
        return ({
            'type': msg.type,
            'dt': msg.time,
            'tempo': msg.tempo})
    if msg.type == 'note_on' or msg.type == 'note_off':
        return ({
            'type': 'note_on',
            'dt': msg.time,
            'key': msg.note-key_offset,
            'vel': msg.velocity if msg.type == 'note_on' else 0})

def event2vec(event):
    '''Transform a single event into a vector of length 260'''
    dt = np.array((event['dt'],)) # list of 0 and 1's
    on_off = np.array((
        1 if event['type'] == 'note_on' else 0,
        1 if event['type'] == 'set_tempo' else 0,)) # onehot encoding for on/off
    if event['type'] == 'set_tempo':
        # binary representation of 
        tempo = np.array((event['tempo'],))
        key = np.zeros((128,))
        vel = np.zeros((128,))
    else:
        # one-hot encode for note on/off
        tempo = np.array((0,))
        key = np.zeros((128,))
        key[event['key']] = 1
        vel = np.zeros((128,))
        vel[event['vel']] = 1
    return np.concatenate((dt, tempo, on_off, key, vel)).astype(np.int32)

def vec2event(vec):
    '''Transform a vector of length 260 into a midi event'''
    dt = vec[0] # get first component of vector
    tempo = vec[1] # get first component of vector
    vec_mode = vec[2:4] # get on/of components [1 hot]
    vec_key = vec[-256:-128] # get key components [1 hot]
    vec_vel = vec[-128:] # get velocity components [1 hot]
    mode = ['note_on', 'set_tempo'][np.argmax(vec_mode)]
    key = np.argmax(vec_key) # get the 1-hot index
    vel = np.argmax(vec_vel) # get the 1-hot index
    return {'type': mode, 'dt': int(dt), 'tempo': int(tempo), 'key': key, 'vel': vel}

def event2msg(event):
    '''Convert an internally used event dict into a mido midi message'''
    if event['type'] == 'set_tempo':
        return MetaMessage(event['type'],
            time=event['dt'],
            tempo=event['tempo'])
    else:
        return Message(event['type'],
            time=event['dt'],
            note=event['key'],
            velocity=event['vel'])

def vec2msg (vec):
    '''Conver a 260 vector to a mido.Message'''
    return event2msg(vec2event(vec))

def translate_file(in_path, out_path, welltempered=False):
    '''Read midi file from in_path and create an np array file at out_path'''
    events = parse_events(in_path)
    if welltempered:
        for k in range(-6,6):
            _events = events.copy()
            for e in _events:
                if 'key' in e:
                    e.update({'key': e['key']+k})
            vecs = np.array([event2vec(e) for e in _events])
            np.save('{}-{}.npy'.format(out_path.rstrip('.npy'),k), vecs)
    else:
        vecs = np.array([event2vec(e) for e in events])
        np.save(out_path, vecs)

def welltempered(events):
    out = None
    for k in range(-6,6,3):
        _events = events.copy()
        for e in _events:
            if 'key' in e:
                e.update({'key': e['key']+k})
        vecs = np.array([event2vec(e) for e in _events])
        if out is None:
            out = vecs
        else:
            np.concatenate((out, vecs), 0)
    return out


def translate_dir(in_dir, out_dir, welltempered=False):
    '''Read all midi files in in_dir and create corrosponding .npy vec files in out_dir'''
    for in_path in glob(in_dir):
        if in_path.lower().endswith('.mid'):
            try:
                print('Processing {}'.format(in_path))
                translate_file(in_path, out_dir+basename(in_path).rstrip('.mid')+'.npy', welltempered=welltempered)
            except Exception as e:
                print('Exception in above file, ',e)
        else:
            print('Skipping: {}'.format(in_path))

def load_back():
    vecs = np.load('test.npy')
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    for vec in vecs:
        msg = event2msg(vec2event(vec))
        track.append(msg)
    mid.save('test.mid')

if __name__ == '__main__':
    # simple tests
    src_dir = 'data/*'
    target_dir = 'proccessed/'
    if len(sys.argv) == 2:
        src_dir = sys.argv[1]
    elif len(sys.argv) == 3:
        src_dir = sys.argv[1]
        target_dir = sys.argv[2]
    else:
        sys.exit('No src dir provided, aborting\nUsage:\n\tpython mido_utils.py <SRC_PATH> <TARGET_PATH?>')
    if not exists(target_dir):
        mkdir(target_dir)
    translate_dir(src_dir, target_dir, welltempered=False)
