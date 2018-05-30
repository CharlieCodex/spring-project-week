from mido import MidiFile, MidiTrack, MetaMessage, Message, merge_tracks
import numpy as np
from os import listdir

def parse_events(f):
    midi = MidiFile(f)
    events = []
    tracks = [midi.tracks[0]]
    tracks.extend([track for track in midi.tracks if 'piano' in track.name.lower()])
    print(tracks)
    for msg in merge_tracks(tracks):
        if msg.type == 'set_tempo':
            events.append({
                'type': msg.type,
                'dt': msg.time,
                'tempo': msg.tempo})
        if msg.type == 'note_on' or msg.type == 'note_off':
            events.append({
                'type': msg.type,
                'dt': msg.time,
                'key': msg.note,
                'vel': msg.velocity})
    return events

def event2vec(event):
    '''Transform a single event into a vector of length 261'''
    dt = np.array((event['dt'],)) # list of 0 and 1's
    on_off = np.array((
        1 if event['type'] == 'note_on' else 0,
        1 if event['type'] == 'note_off' else 0,
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
    '''Transform a vector of length 261 into a midi event'''
    # print(vec)
    dt = vec[0] # get first component of vector
    tempo = vec[1] # get first component of vector
    vec_mode = vec[2:5] # get on/of components [1 hot]
    vec_key = vec[-256:-128] # get key components [1 hot]
    vec_vel = vec[-128:] # get velocity components [1 hot]
    mode = ['note_on', 'note_off', 'set_tempo'][np.argmax(vec_mode)]
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
    '''Conver a 261 vector to a mido.Message'''
    return event2msg(vec2event(vec))

def translate_file(in_path, out_path):
    '''Read midi file from in_path and create an np array file at out_path'''
    events = parse_events(in_path)
    vecs = np.array([event2vec(e) for e in events])
    np.save(out_path, vecs)

def translate_dir(in_dir, out_dir):
    '''Read all midi files in in_dir and create corrosponding .npy vec files in out_dir'''
    for in_path in listdir(in_dir):
        if in_path.endswith('.mid'):
            try:
                print('Processing {}'.format(in_path))
                translate_file(in_dir+in_path, out_dir+in_path.rstrip('.mid')+'.npy')
            except Exception as e:
                print(e)
        else:
            print('Skipping: {}'.format(in_path))

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

if __name__ == '__main__':
    # simple tests
    translate_dir('data pt2/', 'processed/')
