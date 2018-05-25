addr = 0x79
def watch_addr(fn):
    def internal(self, *args, **kwargs):
        if self.offset == addr:
            print('we out here')
        return fn(self, *args, **kwargs)
    return internal

class CountedBuffer:
    def __init__(self, raw):
        self.raw = raw
        self.offset = 0

    def read_int(self, n_bytes):
        self.offset += n_bytes
        return int.from_bytes(self.raw.read(n_bytes), 'big')

    def _read_var_length_bits(self):
        bits = format(int.from_bytes(self.raw.read(1),'big'), '08b')
        n = 0
        if bits[0] == '1':
            n_bits, n = self._read_var_length_bits()
            bits = bits[1:] + n_bits
        return bits[1:], n+1

    def read_var_length(self):
        bits, n = self._read_var_length_bits()
        self.offset += n 
        return int(bits,2), n
    @watch_addr
    def read_hex(self, n_bytes):
        self.offset += n_bytes
        return self.raw.read(n_bytes).hex()

    def read(self, n_bytes):
        self.offset += n_bytes
        return self.raw.read(n_bytes)

    def peek(self, n_bytes):
        return self.raw.peek(n_bytes)

    def skip(self, n_bytes):
        self.offset += n_bytes
        self.raw.read(n_bytes)

    def skip_until_break(self, break_code):
        n=0
        while self.raw.read(1) != break_code:
            n+=1
        self.offset += n
        return n            

def read_header(buf):
    string_lit = buf.read(4)
    if string_lit != b'MThd':
        raise(ValueError('read_header called on non-header chunk'))
    length = buf.read_int(4)
    if length != 6:
        raise(ValueError(
            'read_header for unknown size/format of {} bytes'.format(
            length
        )))
    fmt = buf.read(2)
    n_tracks = buf.read(2)
    division = buf.read(2)
    print('{:#04x}'.format(buf.offset))
    return {'fmt': fmt, 'tracks': n_tracks, 'div': division}

def read_chunk(buf):
    print('new block at {:#04x}'.format(buf.offset))
    events = []
    string_lit = buf.read(4)
    if string_lit == b'MTrk':
        length = buf.read_int(4)
        end_offset = buf.offset + length
        while buf.offset < end_offset:
            dt, _ = buf.read_var_length()
            print('dt', dt)
            print('bytes left in chunk: ', end_offset-buf.offset)
            print('reading until {:#04x}'.format(end_offset))
            desc = buf.read_int(1)
            e_descriptor = (desc>>4, desc%0x10)
            print('Descriptor: ',e_descriptor)
            if e_descriptor[0] == 0xF:
                if e_descriptor[1] == 0xF:
                    # meta event
                    meta_type = buf.read_hex(1)
                    meta_length, _ = buf.read_var_length()
                    buf.skip(meta_length)
                    print('\tMeta event {}, {} bytes'.format(meta_type, meta_length))
                else:
                    # sysex event, format 1
                    sysex_length, _ = buf.read_var_length()
                    buf.skip(sysex_length)
                    print('\tSysex event {:02x}, {} bytes'.format(e_descriptor[0]*0x10 + e_descriptor[1], sysex_length))
            elif e_descriptor[0] == 0x8:
                key = buf.read_int(1)
                vel = buf.read_int(1)
                events.append({'type':'note_off', 'key':key, 'vel':vel})
            elif e_descriptor[0] == 0x9:
                key = buf.read_int(1)
                vel = buf.read_int(1)
                events.append({'type':'note_on', 'key':key, 'vel':vel})
            elif 0xA <= e_descriptor[0] <= 0xB or e_descriptor == 0xE:
                print('\tskipping A, B, or E')
                buf.skip(2)
            elif 0xC <= e_descriptor[0] <= 0xD:
                print('\tskipping C or D')
                buf.skip(1)
            else:
                print('\tUnknown code encountered: {:01x}{:01x}'.format(e_descriptor[0], e_descriptor[1]))
                buf.skip(end_offset - buf.offset)
            print('\toffset {:02x}'.format(buf.offset))
    else:
        print('Unknown block descriptor {}'.format(string_lit))
        raise(ValueError('Invalid block descriptor at {:#x}'.format(buf.offset)))
    return events
            

if __name__ == '__main__':
    buf = CountedBuffer(open('sample.mid','rb'))
    header = read_header(buf)
    events = []
    while buf.peek(4) == b'MThd':
        events.append(read_chunk(buf))