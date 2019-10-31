# explore data
import os, midi, math


# INDICES IN BATCHES (LENGTH,FREQ,VELOCITY are repeated tones_per_cell times):
TICKS_FROM_PREV_START = 0
LENGTH     = 1
FREQ       = 2
VELOCITY   = 3

# INDICES IN SONG DATA (NOT YET BATCHED):
BEGIN_TICK = 0

NUM_FEATURES_PER_TONE = 3

output_ticks_per_quarter_note = 384.0
tones_per_cell=1

def tone_to_freq(tone):
  """
    returns the frequency of a tone. 

    formulas from
      * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
      * https://en.wikipedia.org/wiki/Cent_(music)
  """
  return math.pow(2, ((float(tone)-69.0)/12.0)) * 440.0


def freq_to_tone(freq):
  """
    returns a dict d where
    d['tone'] is the base tone in midi standard
    d['cents'] is the cents to make the tone into the exact-ish frequency provided.
               multiply this with 8192 to get the midi pitch level.

    formulas from
      * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
      * https://en.wikipedia.org/wiki/Cent_(music)
  """
  if freq <= 0.0:
    return None
  float_tone = (69.0+12*math.log(float(freq)/440.0, 2))
  int_tone = int(float_tone)
  cents = int(1200*math.log(float(freq)/tone_to_freq(int_tone), 2))
  return {'tone': int_tone, 'cents': cents}


def read_one_file(path, filename):
    try:
      midi_pattern = midi.read_midifile(os.path.join(path,filename))
    except:
      print ( 'Error reading {}'.format(os.path.join(path,filename)))
      return None
    #
    # Interpreting the midi pattern.
    # A pattern has a list of tracks
    # (midi.Track()).
    # Each track is a list of events:
    #   * midi.events.SetTempoEvent: tick, data([int, int, int])
    #     (The three ints are really three bytes representing one integer.)
    #   * midi.events.TimeSignatureEvent: tick, data([int, int, int, int])
    #     (ignored)
    #   * midi.events.KeySignatureEvent: tick, data([int, int])
    #     (ignored)
    #   * midi.events.MarkerEvent: tick, text, data
    #   * midi.events.PortEvent: tick(int), data
    #   * midi.events.TrackNameEvent: tick(int), text(string), data([ints])
    #   * midi.events.ProgramChangeEvent: tick, channel, data
    #   * midi.events.ControlChangeEvent: tick, channel, data
    #   * midi.events.PitchWheelEvent: tick, data(two bytes, 14 bits)
    #
    #   * midi.events.NoteOnEvent:  tick(int), channel(int), data([int,int]))
    #     - data[0] is the note (0-127)
    #     - data[1] is the velocity.
    #     - if velocity is 0, this is equivalent of a midi.NoteOffEvent
    #   * midi.events.NoteOffEvent: tick(int), channel(int), data([int,int]))
    #
    #   * midi.events.EndOfTrackEvent: tick(int), data()
    #
    # Ticks are relative.
    #
    # Tempo are in microseconds/quarter note.
    #
    # This interpretation was done after reading
    # http://electronicmusic.wikia.com/wiki/Velocity
    # http://faydoc.tripod.com/formats/mid.htm
    # http://www.lastrayofhope.co.uk/2009/12/23/midi-delta-time-ticks-to-seconds/2/
    # and looking at some files. It will hopefully be enough
    # for the use in this project.
    #
    # We'll save the data intermediately with a dict representing each tone.
    # The dicts we put into a list. Times are microseconds.
    # Keys: 'freq', 'velocity', 'begin-tick', 'tick-length'
    #
    # 'Output ticks resolution' are fixed at a 32th note,
    #   - so 8 ticks per quarter note.
    #
    # This approach means that we do not currently support
    #   tempo change events.
    #
    # TODO 1: Figure out pitch.ftone
    # TODO 2: Figure out different channels and instruments.
    #
    
    song_data = []
    
    # Tempo:
    ticks_per_quarter_note = float(midi_pattern.resolution)
    #print (('Resoluton: {}'.format(ticks_per_quarter_note))
    input_ticks_per_output_tick = ticks_per_quarter_note/output_ticks_per_quarter_note
    
    # Multiply with output_ticks_pr_input_tick for output ticks.
    for track in midi_pattern:
      last_event_input_tick=0
      not_closed_notes = []
      for event in track:
        if type(event) == midi.events.SetTempoEvent:
          pass # These are currently ignored
        elif (type(event) == midi.events.NoteOffEvent) or \
             (type(event) == midi.events.NoteOnEvent and \
              event.velocity == 0):
          retained_not_closed_notes = []
          for e in not_closed_notes:
            if tone_to_freq(event.data[0]) == e[FREQ]:
              event_abs_tick = float(event.tick+last_event_input_tick)/input_ticks_per_output_tick
              #current_note['length'] = float(ticks*microseconds_per_tick)
              e[LENGTH] = event_abs_tick-e[BEGIN_TICK]
              song_data.append(e)
            else:
              retained_not_closed_notes.append(e)

          not_closed_notes = retained_not_closed_notes
        elif type(event) == midi.events.NoteOnEvent:
          begin_tick = float(event.tick+last_event_input_tick)/input_ticks_per_output_tick
          note = [0.0]*(NUM_FEATURES_PER_TONE+1)
          note[FREQ]       = tone_to_freq(event.data[0])
          note[VELOCITY]   = float(event.data[1])
          note[BEGIN_TICK] = begin_tick
          not_closed_notes.append(note)
          #not_closed_notes.append([0.0, tone_to_freq(event.data[0]), velocity, begin_tick, event.channel])
        last_event_input_tick += event.tick
      for e in not_closed_notes:
        #print (('Warning: found no NoteOffEvent for this note. Will close it. {}'.format(e))
        e[LENGTH] = float(ticks_per_quarter_note)/input_ticks_per_output_tick
        song_data.append(e)
    song_data.sort(key=lambda e: e[BEGIN_TICK])

    return song_data


def get_midi_pattern(song_data, bpm, tick_is_relative):
    """
    get_midi_pattern takes a song in internal representation 
    (a tensor of dimensions [songlength, self.num_song_features]).
    the three values are length, frequency, velocity.
    if velocity of a frame is zero, no midi event will be
    triggered at that frame.

    returns the midi_pattern.

    Can be used with filename == None. Then nothing is saved, but only returned.
    """

    #
    # Interpreting the midi pattern.
    # A pattern has a list of tracks
    # (midi.Track()).
    # Each track is a list of events:
    #   * midi.events.SetTempoEvent: tick, data([int, int, int])
    #     (The three ints are really three bytes representing one integer.)
    #   * midi.events.TimeSignatureEvent: tick, data([int, int, int, int])
    #     (ignored)
    #   * midi.events.KeySignatureEvent: tick, data([int, int])
    #     (ignored)
    #   * midi.events.MarkerEvent: tick, text, data
    #   * midi.events.PortEvent: tick(int), data
    #   * midi.events.TrackNameEvent: tick(int), text(string), data([ints])
    #   * midi.events.ProgramChangeEvent: tick, channel, data
    #   * midi.events.ControlChangeEvent: tick, channel, data
    #   * midi.events.PitchWheelEvent: tick, data(two bytes, 14 bits)
    #
    #   * midi.events.NoteOnEvent:  tick(int), channel(int), data([int,int]))
    #     - data[0] is the note (0-127)
    #     - data[1] is the velocity.
    #     - if velocity is 0, this is equivalent of a midi.NoteOffEvent
    #   * midi.events.NoteOffEvent: tick(int), channel(int), data([int,int]))
    #
    #   * midi.events.EndOfTrackEvent: tick(int), data()
    #
    # Ticks are relative.
    #
    # Tempo are in microseconds/quarter note.
    #
    # This interpretation was done after reading
    # http://electronicmusic.wikia.com/wiki/Velocity
    # http://faydoc.tripod.com/formats/mid.htm
    # http://www.lastrayofhope.co.uk/2009/12/23/midi-delta-time-ticks-to-seconds/2/
    # and looking at some files. It will hopefully be enough
    # for the use in this project.
    #
    # This approach means that we do not currently support
    #   tempo change events.
    #
    
    # Tempo:
    # Multiply with output_ticks_pr_input_tick for output ticks.
    midi_pattern = midi.Pattern([], resolution=int(output_ticks_per_quarter_note))
    cur_track = midi.Track([])
    cur_track.append(midi.events.SetTempoEvent(tick=0, bpm=bpm))
    future_events = {}
    last_event_tick = 0
    
    ticks_to_this_tone = 0.0
    song_events_absolute_ticks = []
    abs_tick_note_beginning = 0.0
    for frame in song_data:
      # to support absolute ticks
      if tick_is_relative:
      	abs_tick_note_beginning += frame[TICKS_FROM_PREV_START]
      else:
      	abs_tick_note_beginning = frame[TICKS_FROM_PREV_START]

      for subframe in range(tones_per_cell):
        offset = subframe*NUM_FEATURES_PER_TONE
        tick_len           = int(round(frame[offset+LENGTH]))
        freq               = frame[offset+FREQ]
        velocity           = min(int(round(frame[offset+VELOCITY])),127)
        
        d = freq_to_tone(freq)
        if d is not None and velocity > 0 and tick_len > 0:
          # range-check with preserved tone, changed one octave:
          tone = d['tone']
          while tone < 0:   tone += 12
          while tone > 127: tone -= 12

          song_events_absolute_ticks.append((abs_tick_note_beginning,
                                             midi.events.NoteOnEvent(
                                                   tick=0,
                                                   velocity=velocity,
                                                   pitch=tone)))
          song_events_absolute_ticks.append((abs_tick_note_beginning+tick_len,
                                             midi.events.NoteOffEvent(
                                                    tick=0,
                                                    velocity=0,
                                                    pitch=tone)))
    song_events_absolute_ticks.sort(key=lambda e: e[0])
    abs_tick_note_beginning = 0.0
    for abs_tick,event in song_events_absolute_ticks:
      rel_tick = abs_tick-abs_tick_note_beginning
      event.tick = int(round(rel_tick))
      cur_track.append(event)
      abs_tick_note_beginning=abs_tick
    
    cur_track.append(midi.EndOfTrackEvent(tick=int(output_ticks_per_quarter_note)))
    midi_pattern.append(cur_track)

    return midi_pattern


def save_midi_pattern(filename, midi_pattern):
	if filename is not None:
	  midi.write_midifile(filename, midi_pattern)


def save_data(filename, song_data, bpm=48, tick_is_relative=True):
	"""
	save_data takes a filename and a song in internal representation 
	(a tensor of dimensions [songlength, 3]).
	the three values are length, frequency, velocity.
	if velocity of a frame is zero, no midi event will be
	triggered at that frame.

	returns the midi_pattern.

	Can be used with filename == None. Then nothing is saved, but only returned.
	"""
	midi_pattern = get_midi_pattern(song_data, bpm=bpm, tick_is_relative=tick_is_relative)
	save_midi_pattern(filename, midi_pattern)
	return midi_pattern
