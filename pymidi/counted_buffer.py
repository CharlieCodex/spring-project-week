
class CountedBuffer:
    def __init__(self, raw):
        self.raw = raw
        self.offset = 0

    def read_int(self, n_bytes):
        '''Reads an n_byte integer from the internal buffer'''
        self.offset += n_bytes
        return int.from_bytes(self.raw.read(n_bytes), 'big')

    def _read_var_length_bits(self):
        '''Reads a midi variable length value and returns a string of ('0','1',)'''
        bits = format(int.from_bytes(self.raw.read(1),'big'), '08b')
        n = 0
        if bits[0] == '1':
            n_bits, n = self._read_var_length_bits()
            bits = bits[1:] + n_bits
        return bits[1:], n+1

    def read_var_length(self):
        '''Reads a midi variable length value and returns an integer'''
        bits, n = self._read_var_length_bits()
        self.offset += n 
        return int(bits,2), n

    def read_hex(self, n_bytes):
        '''Reads n_bytes and returns a string in hexidecimal'''
        self.offset += n_bytes
        return self.raw.read(n_bytes).hex()

    def read(self, n_bytes):
        self.offset += n_bytes
        return self.raw.read(n_bytes)

    def peek(self, n_bytes):
        return self.raw.peek(n_bytes)[:n_bytes]

    def skip(self, n_bytes):
        self.offset += n_bytes
        self.raw.read(n_bytes)

    def skip_until_break(self, break_code):
        n=0
        while self.raw.read(1) != break_code:
            n+=1
        self.offset += n
        return n            
