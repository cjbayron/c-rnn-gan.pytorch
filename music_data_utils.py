# Tools to load and save midi files for the rnn-gan-project.
# 
# This file has been modified by Christopher John Bayron to support
# operations in c-rnn-gan.pytorch project. 
#
# Written by Olof Mogren, http://mogren.one/
#
# This file has been modified by Christopher John Bayron to support
# c-rnn-gan.pytorch operations. Original file is available in:
#
#     https://github.com/olofmogren/c-rnn-gan
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os, midi, math, random, re, sys
import numpy as np
from io import BytesIO

GENRE      = 0
COMPOSER   = 1
SONG_DATA  = 2

# INDICES IN BATCHES (LENGTH,TONE,VELOCITY are repeated self.tones_per_cell times):
TICKS_FROM_PREV_START      = 0
LENGTH     = 1
TONE       = 2
VELOCITY   = 3

# INDICES IN SONG DATA (NOT YET BATCHED):
BEGIN_TICK = 0

NUM_FEATURES_PER_TONE = 3
IDEAL_TEMPO = 120.0

# hand-picked values for normalization
# "NZ" = "normalizer"
NZ = {
  TICKS_FROM_PREV_START: {'u': 60.0, 's': 80.0},
  LENGTH:                {'u': 64.0, 's': 64.0},
  TONE:                  {'min': 0, 'max': 127},
  VELOCITY:              {'u': 64.0, 's': 128.0},
}

debug = ''
#debug = 'overfit'

sources                              = {}
sources['classical']                 = {}

file_list = {}

file_list['validation'] = [
'classical/sonata-ish/mozk333c.mid', \
'classical/sonata-ish/mozk331b.mid', \
'classical/sonata-ish/mozk313a.mid', \
'classical/sonata-ish/mozk310b.mid', \
'classical/sonata-ish/mozk299a.mid', \
'classical/sonata-ish/mozk622c.mid', \
'classical/sonata-ish/mozk545b.mid', \
'classical/sonata-ish/mozk299a.mid'
]

file_list['test'] = []


# normalization, de-normalization functions
def norm_std(batch_songs, ix):
  vals = batch_songs[:, :, ix]
  vals = (vals - NZ[ix]['u']) / NZ[ix]['s']
  batch_songs[:, :, ix] = vals

def norm_minmax(batch_songs, ix):
  ''' Min-max normalization, to range: [-1, 1]
  '''
  vals = batch_songs[:, :, ix]
  vals = 2*((vals - NZ[ix]['min']) / (NZ[ix]['max'] - NZ[ix]['min'])) - 1
  batch_songs[:, :, ix] = vals

def de_norm_std(song_data, ix):
  vals = song_data[:, ix]
  vals = (vals * NZ[ix]['s']) + NZ[ix]['u']
  song_data[:, ix] = vals

def de_norm_minmax(song_data, ix):
  vals = song_data[:, ix]
  vals = ((vals + 1) / 2)*(NZ[ix]['max'] - NZ[ix]['min']) + NZ[ix]['min']
  song_data[:, ix] = vals


class MusicDataLoader(object):

  def __init__(self, datadir, pace_events=False, tones_per_cell=1, single_composer=None):
    self.datadir = datadir
    self.output_ticks_per_quarter_note = 384.0
    self.tones_per_cell = tones_per_cell
    self.single_composer = single_composer
    self.pointer = {}
    self.pointer['validation'] = 0
    self.pointer['test'] = 0
    self.pointer['train'] = 0
    if not datadir is None:
      print ('Data loader: datadir: {}'.format(datadir))
      self.read_data(pace_events)


  def read_data(self, pace_events):
    """
    read_data takes a datadir with genre subdirs, and composer subsubdirs
    containing midi files, reads them into training data for an rnn-gan model.
    Midi music information will be real-valued frequencies of the
    tones, and intensity taken from the velocity information in
    the midi files.

    returns a list of tuples, [genre, composer, song_data]
    Also saves this list in self.songs.

    Time steps will be fractions of beat notes (32th notes).
    """

    self.genres = sorted(sources.keys())
    print (('num genres:{}'.format(len(self.genres))))
    if self.single_composer is not None:
      self.composers = [self.single_composer]
    else:
      self.composers = []
      for genre in self.genres:
        self.composers.extend(sources[genre].keys())
      if debug == 'overfit':
        self.composers = self.composers[0:1]
      self.composers = list(set(self.composers))
      self.composers.sort()
    print (('num composers: {}'.format(len(self.composers))))

    self.songs = {}
    self.songs['validation'] = []
    self.songs['test'] = []
    self.songs['train'] = []

    # OVERFIT
    count = 0

    for genre in self.genres:
      # OVERFIT
      if debug == 'overfit' and count > 20: break
      for composer in self.composers:
        # OVERFIT
        if debug == 'overfit' and composer not in self.composers: continue
        if debug == 'overfit' and count > 20: break
        current_path = os.path.join(self.datadir,os.path.join(genre, composer))
        if not os.path.exists(current_path):
          print ( 'Path does not exist: {}'.format(current_path))
          continue
        files = os.listdir(current_path)
        #composer_id += 1
        #if composer_id > max_composers:
        #  print (('Only using {} composers.'.format(max_composers))
        #  break
        for i,f in enumerate(files):
          # OVERFIT
          if debug == 'overfit' and count > 20: break
          count += 1
          
          if i % 100 == 99 or i+1 == len(files):
            print ( 'Reading files {}/{}: {}'.format(genre, composer, (i+1)))
          if os.path.isfile(os.path.join(current_path,f)):
            song_data = self.read_one_file(current_path, f, pace_events)
            if song_data is None:
              continue
            if os.path.join(os.path.join(genre, composer), f) in file_list['validation']:
              self.songs['validation'].append([genre, composer, song_data])
            elif os.path.join(os.path.join(genre, composer), f) in file_list['test']:
              self.songs['test'].append([genre, composer, song_data])
            else:
              self.songs['train'].append([genre, composer, song_data])

    random.shuffle(self.songs['train'])
    self.pointer['validation'] = 0
    self.pointer['test'] = 0
    self.pointer['train'] = 0
    # DEBUG: OVERFIT. overfit.
    if debug == 'overfit':
      self.songs['train'] = self.songs['train'][0:1]
      #print (('DEBUG: trying to overfit on the following (repeating for train/validation/test):')
      for i in range(200):
        self.songs['train'].append(self.songs['train'][0])
      self.songs['validation'] = self.songs['train'][0:1]
      self.songs['test'] = self.songs['train'][0:1]
    #print (('lens: train: {}, val: {}, test: {}'.format(len(self.songs['train']), len(self.songs['validation']), len(self.songs['test'])))
    return self.songs

  def read_one_file(self, path, filename, pace_events):
    try:
      if debug:
        print (('Reading {}'.format(os.path.join(path,filename))))
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
    # TODO 1: Figure out pitch.
    # TODO 2: Figure out different channels and instruments.
    #
    
    song_data = []
    tempos = []
    
    # Tempo:
    ticks_per_quarter_note = float(midi_pattern.resolution)
    #print (('Resoluton: {}'.format(ticks_per_quarter_note))
    input_ticks_per_output_tick = ticks_per_quarter_note/self.output_ticks_per_quarter_note
    #if debug == 'overfit': input_ticks_per_output_tick = 1.0
    
    # Multiply with output_ticks_pr_input_tick for output ticks.
    for track in midi_pattern:
      last_event_input_tick=0
      not_closed_notes = []
      for event in track:
        if type(event) == midi.events.SetTempoEvent:
          td = event.data # tempo data
          tempo = 60 * 1000000 / (td[0]*(256**2) + td[1]*256 + td[2])
          tempos.append(tempo)

        elif (type(event) == midi.events.NoteOffEvent) or \
             (type(event) == midi.events.NoteOnEvent and \
              event.velocity == 0):
          retained_not_closed_notes = []
          for e in not_closed_notes:
            if event.data[0] == e[TONE]:
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
          note[TONE]       = event.data[0]
          note[VELOCITY]   = float(event.data[1])
          note[BEGIN_TICK] = begin_tick
          not_closed_notes.append(note)
          
        last_event_input_tick += event.tick
      for e in not_closed_notes:
        #print (('Warning: found no NoteOffEvent for this note. Will close it. {}'.format(e))
        e[LENGTH] = float(ticks_per_quarter_note)/input_ticks_per_output_tick
        song_data.append(e)
    song_data.sort(key=lambda e: e[BEGIN_TICK])
    if (pace_events):
      pace_event_list = []
      pace_tick = 0.0
      song_tick_length = song_data[-1][BEGIN_TICK]+song_data[-1][LENGTH]
      while pace_tick < song_tick_length:
        song_data.append([0.0, 440.0, 0.0, pace_tick, 0.0])
        pace_tick += float(ticks_per_quarter_note)/input_ticks_per_output_tick
      song_data.sort(key=lambda e: e[BEGIN_TICK])

    # tick adjustment (based on tempo)
    avg_tempo = sum(tempos) / len(tempos)
    for frame in song_data:
      frame[BEGIN_TICK] = frame[BEGIN_TICK] * IDEAL_TEMPO/avg_tempo

    return song_data

  def rewind(self, part='train'):
    self.pointer[part] = 0

  def get_batch(self, batchsize, songlength, part='train', normalize=True):
    """
      get_batch() returns a batch from self.songs, as a
      pair of tensors (genrecomposer, song_data).
      
      The first tensor is a tensor of genres and composers
        (as two one-hot vectors that are concatenated).
      The second tensor contains song data.
        Song data has dimensions [batchsize, songlength, num_song_features]

      To have the sequence be the primary index is convention in
      tensorflow's rnn api.
      The tensors will have to be split later.
      Songs are currently chopped off after songlength.
      TODO: handle this in a better way.

      Since self.songs was shuffled in read_data(), the batch is
      a random selection without repetition.

      songlength is related to internal sample frequency.
      We fix this to be every 32th notes. # 50 milliseconds.
      This means 8 samples per quarter note.
      There is currently no notion of tempo in the representation.

      composer and genre is concatenated to each event
      in the sequence. There might be more clever ways
      of doing this. It's not reasonable to change composer
      or genre in the middle of a song.
      
      A tone  has a feature telling us the pause before it.

    """
    #print (('get_batch(): pointer: {}, len: {}, batchsize: {}'.format(self.pointer[part], len(self.songs[part]), batchsize))
    if self.pointer[part] > len(self.songs[part])-batchsize:
      batchsize = len(self.songs[part]) - self.pointer[part]
      if batchsize == 0:
      	return None, None

    if self.songs[part]:
      batch = self.songs[part][self.pointer[part]:self.pointer[part]+batchsize]
      self.pointer[part] += batchsize
      # subtract two for start-time and channel, which we don't include.
      num_meta_features = len(self.genres)+len(self.composers)
      # All features except timing are multiplied with tones_per_cell (default 1)
      num_song_features = NUM_FEATURES_PER_TONE*self.tones_per_cell+1
      batch_genrecomposer = np.ndarray(shape=[batchsize, num_meta_features])
      batch_songs = np.ndarray(shape=[batchsize, songlength, num_song_features])

      for s in range(len(batch)):
        songmatrix = np.ndarray(shape=[songlength, num_song_features])
        composeronehot = onehot(self.composers.index(batch[s][1]), len(self.composers))
        genreonehot = onehot(self.genres.index(batch[s][0]), len(self.genres))
        genrecomposer = np.concatenate([genreonehot, composeronehot])
        
        #random position:
        begin = 0
        if len(batch[s][SONG_DATA]) > songlength*self.tones_per_cell:
          begin = random.randint(0, len(batch[s][SONG_DATA])-songlength*self.tones_per_cell)
        matrixrow = 0
        n = begin
        while matrixrow < songlength:
          eventindex = 0
          event = np.zeros(shape=[num_song_features])
          if n < len(batch[s][SONG_DATA]):
            event[LENGTH]   = batch[s][SONG_DATA][n][LENGTH]
            event[TONE]     = batch[s][SONG_DATA][n][TONE]
            event[VELOCITY] = batch[s][SONG_DATA][n][VELOCITY]
            ticks_from_start_of_prev_tone = 0.0
            if n>0:
              # beginning of this tone, minus starting of previous
              ticks_from_start_of_prev_tone = batch[s][SONG_DATA][n][BEGIN_TICK]-batch[s][SONG_DATA][n-1][BEGIN_TICK]
              # we don't include start-time at index 0:
              # and not channel at -1.
            # tones are allowed to overlap. This is indicated with
            # relative time zero in the midi spec.
            event[TICKS_FROM_PREV_START] = ticks_from_start_of_prev_tone
            tone_count = 1
            for simultaneous in range(1,self.tones_per_cell):
              if n+simultaneous >= len(batch[s][SONG_DATA]):
                break
              if batch[s][SONG_DATA][n+simultaneous][BEGIN_TICK]-batch[s][SONG_DATA][n][BEGIN_TICK] == 0:
                offset = simultaneous*NUM_FEATURES_PER_TONE
                event[offset+LENGTH]   = batch[s][SONG_DATA][n+simultaneous][LENGTH]
                event[offset+TONE]     = batch[s][SONG_DATA][n+simultaneous][TONE]
                event[offset+VELOCITY] = batch[s][SONG_DATA][n+simultaneous][VELOCITY]
                tone_count += 1
              else:
                break
          songmatrix[matrixrow,:] = event
          matrixrow += 1
          n += tone_count
        #if s == 0 and self.pointer[part] == batchsize:
        #  print ( songmatrix[0:10,:]
        batch_genrecomposer[s,:] = genrecomposer
        batch_songs[s,:,:] = songmatrix

      # input normalization
      if normalize:
        norm_std(batch_songs, TICKS_FROM_PREV_START)
        norm_std(batch_songs, LENGTH)
        norm_std(batch_songs, VELOCITY)
        norm_minmax(batch_songs, TONE)

      return batch_genrecomposer, batch_songs

    else:
      raise 'get_batch() called but self.songs is not initialized.'
  
  def get_num_song_features(self):
    return NUM_FEATURES_PER_TONE*self.tones_per_cell+1
  def get_num_meta_features(self):
    return len(self.genres)+len(self.composers)

  def get_midi_pattern(self, song_data, bpm, normalized=True):
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
    midi_pattern = midi.Pattern([], resolution=int(self.output_ticks_per_quarter_note))
    cur_track = midi.Track([])
    cur_track.append(midi.events.SetTempoEvent(tick=0, bpm=IDEAL_TEMPO))
    future_events = {}
    last_event_tick = 0
    
    ticks_to_this_tone = 0.0
    song_events_absolute_ticks = []
    abs_tick_note_beginning = 0.0

    if type(song_data) != np.ndarray:
      song_data = np.array(song_data)

    # de-normalize
    if normalized:
      de_norm_std(song_data, TICKS_FROM_PREV_START)
      de_norm_std(song_data, LENGTH)
      de_norm_std(song_data, VELOCITY)
      de_norm_minmax(song_data, TONE)

    for frame in song_data:
      abs_tick_note_beginning += int(round(frame[TICKS_FROM_PREV_START]))
      for subframe in range(self.tones_per_cell):
        offset = subframe*NUM_FEATURES_PER_TONE
        tick_len           = int(round(frame[offset+LENGTH]))
        tone               = int(round(frame[offset+TONE]))
        velocity           = min(int(round(frame[offset+VELOCITY])),127)
        
        if tone is not None and velocity > 0 and tick_len > 0:
          # range-check with preserved tone, changed one octave:
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
    
    cur_track.append(midi.EndOfTrackEvent(tick=int(self.output_ticks_per_quarter_note)))
    midi_pattern.append(cur_track)

    return midi_pattern

  def save_midi_pattern(self, filename, midi_pattern):
    if filename is not None:
      midi.write_midifile(filename, midi_pattern)

  def save_data(self, filename, song_data, bpm=IDEAL_TEMPO):
    """
    save_data takes a filename and a song in internal representation 
    (a tensor of dimensions [songlength, 3]).
    the three values are length, frequency, velocity.
    if velocity of a frame is zero, no midi event will be
    triggered at that frame.

    returns the midi_pattern.

    Can be used with filename == None. Then nothing is saved, but only returned.
    """
    midi_pattern = self.get_midi_pattern(song_data, bpm=bpm)
    self.save_midi_pattern(filename, midi_pattern)
    return midi_pattern

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

def onehot(i, length):
  a = np.zeros(shape=[length])
  a[i] = 1
  return a
