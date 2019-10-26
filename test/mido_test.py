'''
mido tester
manipulating MIDI files with python
'''

import mido
from argparse import ArgumentParser

def prepare_tracked_sequence(mid):
    
    # by default time info is number of ticks
    messages = []
    for i, track in enumerate(mid.tracks):
        # this replaces time w/ accumulated ticks
        messages.extend(mido.midifiles.tracks._to_abstime(track))

    # sort according to time (accumulated ticks /
    # no. of ticks passed before message event)
    messages.sort(key=lambda msg: msg.time)
    tracks = mido.midifiles.tracks.fix_end_of_track(mido.midifiles.tracks._to_reltime(messages))

    ###############
    # get total time
    time_total = 0
    tempo = 461538
    for msg in tracks:
        delta = 0
        if msg.time > 0:
            delta = mido.midifiles.units.tick2second(msg.time, mid.ticks_per_beat, tempo)

        time_total += delta

    print(time_total)

    exit()
    ###############

    if mid.type == 2:
        raise TypeError("can't merge tracks in type 2 (asynchronous) file")

    # for msg in tracks:
    #     if not msg.is_meta:
    #         print(msg.channel)

    # input()

    tempo = mido.midifiles.midifiles.DEFAULT_TEMPO
    for msg in tracks:
        # Convert message time from absolute time
        # in ticks to relative time in seconds.
        if msg.time > 0:
            delta = mido.midifiles.units.tick2second(msg.time, mid.ticks_per_beat, tempo)
        else:
            delta = 0

        yield msg.copy(time=delta)

        if msg.type == 'set_tempo':
            tempo = msg.tempo

def get_msg(mid):

    # for msg in prepare_tracked_sequence(mid):
    #     print(msg)

    for i, track in enumerate(mid.tracks):
        for msg in track:
            print(msg)

def extract_track(mid):
    # save MIDI only with bass and headers
    TRACK_HEADER = 0
    TRACK_BASS = 2

    bass_mid = mido.MidiFile(type=1)
    bass_mid.ticks_per_beat = mid.ticks_per_beat

    # get tracks to put in new MIDI file
    for i, track in enumerate(mid.tracks):
        if i == TRACK_HEADER:
            bass_mid.tracks.append(track)

        elif i == TRACK_BASS:
            #bass_mid.tracks[0].extend(track)
            new_track = track.copy()
            for msg in new_track:
                if msg.type == 'program_change':
                    msg.program = 1
                    continue

                if msg.type != 'note_on':
                    new_track.remove(msg)
                else:
                    msg.note += 5

            bass_mid.tracks[0].extend(new_track)

    # for msg in mid.tracks[0]:
    #     if msg.type == 'program_change':
    #         msg.program = 1
    #         break

    bass_mid.save('test_bass2.mid')      


if __name__ == '__main__':
    # Command line arguments
    ARG_PARSER = ArgumentParser()
    ARG_PARSER.add_argument('-m', '--mode', dest='mode',
                            required=True,
                            choices=['1', '2'],
                            help='1: messages, 2: extract track')
    ARG_PARSER.add_argument('-i', '--input', dest='input',
                            required=True,
                            help='MIDI file path')

    ARGS = ARG_PARSER.parse_args()

    mid = mido.MidiFile(ARGS.input)

    if(ARGS.mode == '1'):
        get_msg(mid)

    elif(ARGS.mode == '2'):
        extract_track(mid)