
import os
import midi
import math
import glob
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import midi_explorer as exp


def tone_to_freq(tone):
  """
    returns the frequency of a tone. 

    formulas from
      * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
      * https://en.wikipedia.org/wiki/Cent_(music)
  """
  return math.pow(2, ((float(tone)-69.0)/12.0)) * 440.0


def check_set_tempos():
    paths = glob.glob('data/classical/sonata-ish/*')
    for path in paths:
        mid = midi.read_midifile(path)

        tempo_event_cnt = 0
        for track in mid:
            for event in track:
                if type(event) == midi.events.SetTempoEvent:
                    tempo_event_cnt += 1

        if (tempo_event_cnt > 1):
            print(path)


def get_bpm(data):
    div = data[0] * (256**2) + data[1]*256 + data[2]
    bpm = 60 * 1000000 / div
    return bpm


def get_avg_tick_interval(track):
    ticks = []
    for event in track:
        if (type(event) == midi.events.NoteOnEvent) and (event.tick != 0):
            ticks.append(event.tick)

    count = Counter(ticks)
    # print(count.most_common(5))

    return (sum(ticks)/len(ticks))


def check_avg_tick_interval_2():
    p1 = 'test/copy_mozk311b.mid'
    p2 = 'test/copy_mozk331b.mid'   

    a = midi.read_midifile(p1)
    b = midi.read_midifile(p2)

    tma = get_bpm(a[0][0].data)
    tmb = get_bpm(b[0][0].data)

    tra = a[0]
    trb = b[0]

    tia = get_avg_tick_interval(tra)
    tib = get_avg_tick_interval(trb)

    # print(tma, tia)
    # print(tmb, tib)


def generate_copies():
    paths = glob.glob('data/classical/sonata-ish/*.mid')
    for path in paths:
        m = exp.read_one_file(os.path.dirname(path), os.path.basename(path))

        fn = os.path.join('test/copies', os.path.basename(path))
        exp.save_data(fn, m, bpm=120, tick_is_relative=False)


def check_avg_tick_interval_all():
    paths = glob.glob('test/copies/*.mid')
    for path in paths:
        m = midi.read_midifile(path)
        bpm = get_bpm(m[0][0].data)
        tick_i = get_avg_tick_interval(m[0])

        print(path, bpm, tick_i, bpm*tick_i)
        # print(tick_i, path)


def analyze_tick():
    paths = glob.glob('test/copies/*.mid')
    tick_collection = {}
    tick_list = []

    for path in tqdm(paths):
        m = midi.read_midifile(path)

        track = m[0]
        for event in track:
            if (type(event) == midi.events.NoteOnEvent) and (event.tick != 0):
                if not event.tick in tick_collection:
                    tick_collection[event.tick] = 0

                tick_collection[event.tick] += 1
                tick_list.append(event.tick)

    # get most frequent ticks
    for i, tick_val in enumerate(sorted(tick_collection, key=tick_collection.get, reverse=True)):
        print(tick_val, tick_collection[tick_val])
        if i == 20:
            break

    # get weighted mean
    w_sum = 0
    total_tick_cnt = 0
    for tick_val, tick_cnt in tick_collection.items():
        w_sum += (tick_val * tick_cnt)
        total_tick_cnt += tick_cnt

    print("w mean: ", w_sum / total_tick_cnt)

    l = np.array(list(tick_collection.keys()))
    print(min(l), max(l))
    print(np.mean(l), np.std(l))
    tick_list = np.array(tick_list)
    print(np.mean(tick_list), np.std(tick_list))
    print(min(tick_list), max(tick_list))

    plt.hist(tick_list[tick_list < 500], bins=100)
    plt.show()

    tick_list = (tick_list - np.mean(tick_list)) / np.std(tick_list)
    print(min(tick_list), max(tick_list))

    l = (l - 60) / np.std(l)
    print(min(l), max(l))

    # plt.bar(list(tick_collection.keys()), tick_collection.values(), color='g')
    # plt.show()

def analyze_tick_len():
    paths = glob.glob('test/copies/*.mid')
    tick_list = []

    for path in tqdm(paths):
        m = midi.read_midifile(path)

        track = m[0]
        for event in track:
            if (type(event) == midi.events.NoteOffEvent) and (event.tick != 0):
                tick_list.append(event.tick)

    for val, cnt in Counter(tick_list).most_common(10):
        print(val, cnt)

    tick_list = np.array(tick_list)
    print(np.mean(tick_list), np.std(tick_list))
    print(min(tick_list), max(tick_list))

    plt.hist(tick_list[tick_list < 400], bins=100)
    plt.show()

    tick_list = (tick_list - np.mean(tick_list)) / np.std(tick_list)
    print(min(tick_list), max(tick_list))


def analyze_note():
    paths = glob.glob('test/copies/*.mid')
    freq_list = []
    v_list = []

    for path in tqdm(paths):
        m = midi.read_midifile(path)

        track = m[0]
        for event in track:
            if (type(event) == midi.events.NoteOnEvent) and (event.tick != 0):
                # freq_list.append(tone_to_freq(event.data[0]))
                freq_list.append(event.data[0])
                v_list.append(event.data[1])

    for val, cnt in Counter(freq_list).most_common(10):
        print(val, cnt)

    freq_list = np.array(freq_list)
    print(max(freq_list), min(freq_list))
    print(np.mean(freq_list), np.std(freq_list))
    freq_list = (freq_list - np.mean(freq_list)) / np.std(freq_list)
    print(min(freq_list), max(freq_list))

    print('---------')
    for val, cnt in Counter(v_list).most_common(10):
        print(val, cnt)

    v_list = np.array(v_list)
    print(max(v_list), min(v_list))
    print(np.mean(v_list), np.std(v_list))
    v_list = (v_list - np.mean(v_list)) / np.std(v_list)
    print(min(v_list), max(v_list))


if __name__ == "__main__":
    # check_set_tempos()
    # check_avg_tick_interval_2()
    # generate_copies()
    # check_avg_tick_interval_all()
    analyze_tick()
    analyze_tick_len()
    # analyze_note()
