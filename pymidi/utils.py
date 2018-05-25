meta_events = {
    '00': 'Sequence number',
    '01': 'Text event',
    '02': 'Copyright notice',
    '03': 'Sequence or track name',
    '04': 'Instrument name',
    '05': 'Lyric text',
    '06': 'Marker text',
    '07': 'Cue point',
    '20': 'Channel Prefix',
    '2f': 'End of track',
    '51': 'Tempo',
    '54': 'SMTPE offset',
    '58': 'Time signature',
    '59': 'Key signature',
    '7f': 'Sequence specific event',
}

def read_header(buf):
    '''Reads a midi header chunk and returns a dict of fmt, trakcs, and div'''
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
    '''Reads a midi track chunk and returns a list of event dicts,
       each containing {dt, type, key, vel}'''
    print('new block at {:#04x}'.format(buf.offset))
    events = []
    string_lit = buf.read(4)
    if string_lit == b'MTrk':
        length = buf.read_int(4)
        print('Track of length:\n\t{}'.format(length))
        end_offset = buf.offset + length
        while buf.offset < end_offset:
            dt, _ = buf.read_var_length()
            print('\n\tdt', dt)
            print('\tbytes left in chunk: ', end_offset-buf.offset)
            print('\treading until {:#04x}'.format(end_offset))
            desc = buf.read_int(1)
            e_descriptor = (desc>>4, desc%0x10)
            if e_descriptor[0] == 0xF:
                if e_descriptor[1] == 0xF:
                    # meta event
                    meta_type = meta_events[buf.read_hex(1)]
                    meta_length, _ = buf.read_var_length()
                    buf.skip(meta_length)
                    print('\tMeta event "{}", {} bytes'.format(meta_type, meta_length))
                else:
                    # sysex event, format 1
                    sysex_length, _ = buf.read_var_length()
                    buf.skip(sysex_length)
                    print('\tSysex event {:02x}, {} bytes'.format(e_descriptor[0]*0x10 + e_descriptor[1], sysex_length))
            elif e_descriptor[0] == 0x8:
                key = buf.read_int(1)
                vel = buf.read_int(1)
                events.append({'dt': dt, 'type':'note_off', 'key':key, 'vel':0x3F})
            elif e_descriptor[0] == 0x9:
                key = buf.read_int(1)
                vel = buf.read_int(1)
                if vel != 0x00:
                    events.append({'dt': dt, 'type':'note_on', 'key':key, 'vel':vel})
                else:
                    events.append({'dt': dt, 'type':'note_off', 'key':key, 'vel':0x3F})
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
