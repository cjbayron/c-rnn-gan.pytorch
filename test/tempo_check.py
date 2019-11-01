
import os
import midi
import glob
from collections import Counter

import midi_explorer as exp

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

		# print(path, bpm, tick_i, bpm*tick_i)
		print(tick_i, path)

if __name__ == "__main__":
	#check_set_tempos()
	# check_avg_tick_interval_2()
	generate_copies()
	check_avg_tick_interval_all()

