# NOTE: run in base directory!

import music_data_utils


DATA_DIR = 'data'
COMPOSER = 'sonata-ish'

BATCH_SIZE = 32
MAX_SEQ_LEN = 200

if __name__ == "__main__":
    dataloader = music_data_utils.MusicDataLoader(DATA_DIR, single_composer=COMPOSER)

    dataloader.rewind(part='train')
    batch_meta, batch_song = dataloader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part='train')

    while batch_meta is not None and batch_song is not None:

        # checking
        dataloader.save_data('sample.mid', batch_song[0])

        break

        # fetch next batch
        batch_meta, batch_song = dataloader.get_batch(BATCH_SIZE, MAX_SEQ_LEN, part='train')