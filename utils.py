import numpy as np
import glob
import sys

def track_seed():
    """Get a standard seed note for random generation.
        
        :return: a vector ready to be sent off to mido_utils' "vec2msg
    """
    dt = np.array((0,))
    tempo = np.array((0,))
    mode = np.zeros((3,))             # make an zero vector of shape (2,)
    mode[2] = 1                       # one hot encode the on component
    note = np.zeros((128,))
    note[0x37] = 1
    vel = np.zeros((128,))
    vel [0x37] = 1
    return np.concatenate((dt, tempo, mode, note, vel,)).astype(int)

def prep_vec(vec):
    """Roll the dice to produce a random integer in the [0..ALPHASIZE] range,
    according to the provided probabilities. If topn is specified, only the
    topn highest probabilities are taken into account.

        :param probabilities: a list of size ALPHASIZE with individual probabilities
        :param topn: the number of highest probabilities to consider. Defaults to all of them.
        :return: a vector ready to be sent off to mido_utils' "vec2msg"
    """
    vec = vec[0]
    dt = np.array((vec[0],))
    tempo = np.array((vec[1],))
    idx_on_off = np.argmax(vec[2:5])    # get the on of bit and argmax
    on_off = np.zeros((3,))             # make an zero vector of shape (2,)
    on_off[idx_on_off] = 1              # one hot encode the on_off component
    note = one_hot_from_probabilities(vec[-256:-128]) # get the note
    vel = one_hot_from_probabilities(vec[-128:])   # get the velocity
    return np.concatenate((dt, tempo, on_off, note, vel,)).astype(int)

def one_hot_from_probabilities(probabilities, topn=128):
    """Take a vector as a list of probabilties for categories and return a onehot encoded vector,
    representing a sampled choice from the probabilities
        :param probabilities: vector of probabilities
        :param topn: limit which values are sampled from
        :return: a one-hot encoded vector with the same length as probabilites"""
    arg_size = len(probabilities)
    p = np.squeeze(probabilities)
    p[np.argsort(p)[:-topn]] = 0
    p = p / np.sum(p)
    i = np.random.choice(arg_size, 1, p=p)[0]
    out = np.zeros(arg_size)
    out[i] = 1
    return out

def rnn_minibatch_sequencer(raw_data, batch_size, sequence_size, nb_epochs):
    """
    Divides the data into batches of sequences so that all the sequences in one batch
    continue in the next batch. This is a generator that will keep returning batches
    until the input data has been seen nb_epochs times. Sequences are continued even
    between epochs, apart from one, the one corresponding to the end of raw_data.
    The remainder at the end of raw_data that does not fit in an full batch is ignored.
        :param raw_data: the training data
        :param batch_size: the size of a training minibatch
        :param sequence_size: the unroll size of the RNN
        :param nb_epochs: number of epochs to train on
        :return:
            x: one batch of training sequences
            y: on batch of target sequences, i.e. training sequences shifted by 1
            epoch: the current epoch number (starting at 0)
    """
    data = np.array(raw_data) # wrap the raw data so we can do numpy stuffs
    data_len = data.shape[0] # get how many data points we have (this is along the first axis)
    # using (data_len-1) because we must provide for the sequence shifted by 1 too
    nb_batches = (data_len - 1) // (batch_size * sequence_size)
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    # this is rounded so that we batch correctly,
    # rounding is done by using our floor quotient (//) nb_batches
    rounded_data_len = nb_batches * batch_size * sequence_size
    # we now truncate the data at the rounded length and reshape it into:
    # batch_size by
    # nb_batches * seq_size by
    # vec line
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size, -1])
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size, -1])
    # the remainder is a generator func which will yield our batches properly
    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            # rolling lets us make sure the rnn state can be kept from epoch to epoch
            x = np.roll(x, -epoch, axis=0)  
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch

def find_track(index, trackranges):
    return next(
        track["name"] for track in trackranges if (track["start"] <= index < track["end"]))

def find_track_index(index, trackranges):
    return next(
        i for i, track in enumerate(trackranges) if (track["start"] <= index < track["end"]))

def read_data_files(directory):
    """Read data files according to the specified glob pattern
    :param directory: for example "data/*.txt"
    :return: training data, list of loaded file names with ranges
    """
    data = [] # init as list but type will mutate to np.ndarray in first iteration
    trackranges = []
    tracklist = glob.glob(directory, recursive=True)
    nfiles = len(tracklist)
    n = 0
    for midifile in tracklist:
        n+=1
        print("\rLoading file {:04d} of {:04d}: {:_<40.40}".format(
            n,
            nfiles,
            midifile), end='')
        start = len(data)
        if len(data) == 0:
            data = np.load(midifile)
        else:
            np.concatenate((data, np.load(midifile),), axis=0)
        end = len(data)
        trackranges.append({"start": start, "end": end, "name": midifile.rsplit("/", 1)[-1]})

    print()
    if len(trackranges) == 0:
        sys.exit("No training data has been found. Aborting.")

    return data, trackranges