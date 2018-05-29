from mido import MidiFile, MidiTrack, Message, merge_tracks
import numpy as np
from os import listdir

def parse_events(f):
    midi = MidiFile(f)
    events = []
    n = 0
    for msg in merge_tracks(midi.tracks):
        if msg.type == 'set_tempo':
            print(msg.time, msg.tempo)
            #print('\ttempo change to {} µspq after {} notes'.format(msg.tempo,n))
            # tempo = msg.tempo
            n=0
        if msg.type == 'note_on' or msg.type == 'note_off':
            print(msg.time)
            n += 1
            events.append({
                'type': msg.type,
                'dt': msg.time,
                'key': msg.note,
                'vel': msg.velocity})
    return events

def msg2vec(msg):
    '''Transform a mido message into a vector of length 259'''
    return event2vec({
        'type': msg.type,
        'dt': msg.time,
        'key': msg.note,
        'vel': msg.velocity})


def event2vec(event):
    '''Transform a single event into a vector of length 259'''
    dt = np.array((event['dt'],)) # list of 0 and 1's
    on_off = np.array((
        1 if event['type'] == 'note_on' else 0,
        1 if event['type'] == 'note_off' else 0,
        1 if event['type'] == 'tempo' else 0,)) # onehot encoding for on/off
    if event['type'] == 'tempo':
        # binary representation of 
        tempo = np.array([int(c) for c in format('032b',event['tempo'])])
        key = tempo[:128]
        vel = tempo[128:]
    else:
        # one-hot encode for note on/off
        key = np.zeros((128,))
        key[event['key']] = 1
        vel = np.zeros((128,))
        vel[event['vel']] = 1
    return np.concatenate((dt, on_off, key, vel))

def vec2event(vec):
    '''Transform a vector of length 259 into a midi event'''
    # print(vec)
    dt = vec[0] # get first component of vector
    vec_on_off = vec[1:3] # get on/of components [1 hot]
    vec_key = vec[3:3+128] # get key components [1 hot]
    vec_vel = vec[3+128:] # get velocity components [1 hot]
    on_off = 'note_off' if vec_on_off[0] == 0 else 'note_on' # extract on/off -ness of note
    key = np.argmax(vec_key) # get the 1-hot index
    vel = np.argmax(vec_vel) # get the 1-hot index
    return {'type': on_off, 'dt': int(dt), 'key': key, 'vel': vel}

def event2msg(event):
    '''Convert an internally used event dict into a mido midi message'''
    return Message(event['type'],
        time=event['dt'],
        note=event['key'],
        velocity=event['vel'])

def vec2msg (vec):
    '''Conver a 259 vector to a mido.Message'''
    return event2msg(vec2event(vec))

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

if __name__ == '__main__':
    # simple tests
    translate_dir('../data/', '../processed/')

def load_back():
    vecs = np.load('test.npy')
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    for vec in vecs:
        msg = event2msg(vec2event(vec))
        #print(vec, msg)
        track.append(msg)
    track.append(Message('note_on', time=0, note=64, velocity=127))
    track.append(Message('note_off', time=128, note=64, velocity=127))
    track.append(Message('note_on', time=64, note=64, velocity=127))
    track.append(Message('note_on', time=128, note=64, velocity=0))
    mid.save('test.mid')
