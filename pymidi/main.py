import numpy as np

class Division:
    def __init__(self, raw_bytes):
        bits = bin(int.from_bytes(raw_bytes,'big'))[2:].zfill(16)
        print(raw_bytes, bits)
        self.fmt = bits[0] == '1'
        if self.fmt:
            self.frames_per_second = -int(bits[1:7],2)
            self.ticks_per_frame = int(bits[8:15],2)
        else:
            self.ticks_per_quarter = int(bits[1:15], 2)

    def __repr__(self):
        if self.fmt:
            return "Division<{} ticks per frame @ {} fps>".format(self.ticks_per_frame.__repr__(), self.frames_per_second)
        else:
            return "Division<{} ticks per quaternote>".format(self.ticks_per_quarter)

class MidiData:
    def __init__(self, fmt, tracks, division, events=[]):
        self.fmt = fmt
        self.tracks = tracks
        self.division = division
        self.events = events
    
    def add_event(self, event):
        self.events.append(event)

class MidiEvent:
    class MidiEventType:
        EVENT_NOTE_OFF = 0
        EVENT_NOTE_ON = 1
        EVENT_POLY_PRESSURE = 2
        EVENT_CONTROLLER_CHANGE = 3
        EVENT_PROGRAM_CHANGE = 4
        EVENT_CHANNEL_PRESSURE = 5
        EVENT_PITCH_BEND = 6
    def __init__(self, dt, event_type, **kwargs):
        self.dt = dt
        self.event_type = event_type
        self.attrs = kwargs

def read_var_length_bits(buf):
    bits = format(int.from_bytes(buf.read(1),'big'), '08b')
    n = 0
    if bits[0] == '1':
        n_bits, n = read_var_length_bits(buf)
        bits = bits[1:] + n_bits
    return bits[1:], n+1


def read_var_length(buf):
    bits, n = read_var_length_bits(buf)
    return int(bits,2), n

def read_int(buf, n_bytes):
    return int.from_bytes(buf.read(n_bytes), 'big')

if __name__ == '__main__':
    f = open('sample.mid','rb')
    try:
        midi_out = None
        while True:
            block_type = f.read(4)
            if block_type == b"MThd":
                size = int.from_bytes(f.read(4), 'big')
                if size == 6:
                    # only accepted format
                    fmt = int.from_bytes(f.read(2), 'big')
                    tracks = int.from_bytes(f.read(2), 'big')
                    division = Division(f.read(2))
                    midi_out = MidiData(fmt, tracks, division)
                else:
                    break
            elif block_type == b"MTrk":
                print('track')
                size = int.from_bytes(f.read(4), 'big')
                dt, offset = read_var_length(f)
                while offset < size:
                    print('{} of {}'.format(offset, size))
                    e_header = format(read_int(f, 2),'02x')
                    offset+= 2
                    e_code, channel = int(e_header[0],16), int(e_header[1],16)
                    print('Code {:1x}'.format(e_code))
                    if e_code == 0x0:
                        f.read(size-offset)
                        offset = size
                    if e_code == 0x3:
                        pass
                    elif e_code == 0x8 or e_code == 0x9:
                        kk = read_int(f,2)
                        vv = read_int(f,2)
                        offset += 4
                        midi_out.add_event(MidiEvent(dt,MidiEvent.MidiEventType.EVENT_NOTE_OFF,key=kk, velocity=vv))
                    elif e_code == 0xB:
                        number, value = read_int(f, 2), read_int(f, 2)
                        offset += 4
                        midi_out.add_event(MidiEvent(dt,MidiEvent.MidiEventType.EVENT_CONTROLLER_CHANGE))
                    elif e_code == 0xF:
                        if channel == 0xF:
                            meta_type = read_int(f, 1)
                            offset += 1
                            length, _ = read_var_length(f)
                            print('Meta event {:02x}, skipping {} bytes'
                                .format(meta_type, length))
                            f.read(length)
                            offset += length + _
                        else:
                            cur_b = f.read(1)
                            while cur_b != bytes([0xF7]):
                                f.read(1)
                                offset += 1
                    else:
                        print('Unknown code {:02x}'.format(e_code))
                        break
            elif block_type == b'':
                break
            else:
                print('Invalid header', block_type)
    finally:
        f.close()