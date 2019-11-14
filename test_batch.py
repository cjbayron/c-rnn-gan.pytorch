# NOTE: run in base directory!
import numpy as np
import matplotlib.pyplot as plt

import music_data_utils

DATA_DIR = 'data'
COMPOSER = 'overfit'

BATCH_SIZE = 16
MAX_SEQ_LEN = 256

# 0: TICKS_FROM_PREV (try to fix to 0 - 255)
# 1: TICK_LEN (try to fix to 0 - 255)
# 2: TONE     (0 - 127)
# 3: VELOCITY (0 - 127)


def as_image(song):
    song[:, 2] *= 2
    song[:, 3] *= 2

    image = song.reshape([32, -1])
    image[image > 255] = 255

    return image


if __name__ == "__main__":
    dataloader = music_data_utils.MusicDataLoader(DATA_DIR, single_composer=COMPOSER)

    dataloader.rewind(part='train')
    batch_meta, batch_song = dataloader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part='train', normalize=False)

    song_cnt = 0
    while batch_meta is not None and batch_song is not None:

        midi_img = as_image(batch_song)
        plt.imshow(midi_img)
        plt.show()

        if len(batch_song) < BATCH_SIZE:
            batch_song = np.repeat(batch_song, int(BATCH_SIZE/len(batch_song)), axis=0)

        # for song in batch_song:
        #     midi_data = dataloader.save_data(None, song)
        #     print(midi_data[0][:16])

        # checking
        # for song in batch_song:
        #     fn = 'test/samples/sample-{}.mid'.format(song_cnt)
        #     dataloader.save_data(fn, song)
        #     print("Saved %s!" % fn)
        #     song_cnt += 1

        # dataloader.save_data('sample.mid', batch_song[0])
        # print(batch_song[0][:16])

        # break

        # fetch next batch
        batch_meta, batch_song = dataloader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part='train')