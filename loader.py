from essentia.standard import (
    MusicExtractor,
    EqloudLoader,
    PredominantPitchMelodia
)
from constants import PATH_TO_DATASET, WAV_SONGS_FILENAME, SONGS_WAV_PATH

Extractor = MusicExtractor()
pitch_extractor = PredominantPitchMelodia(frameSize=2048, hopSize=128)

__all__ = ["load_all_songs_pitch_vectors"]


def _read_songs_names():
    path = '{}{}'.format(PATH_TO_DATASET, WAV_SONGS_FILENAME)
    file = open(path, 'r')
    songs_wav_paths = [
        '{}/{}'.format(SONGS_WAV_PATH, name.rstrip('\n'))
        for name in file.readlines()
    ]
    file.close()
    return songs_wav_paths


def _load_song(filepath):
    # TODO: Calculate some features in other function, not all of them
    # features, features_frames = Extractor.compute(filepath)
    # features_keys = features.descriptorNames()

    # Loads the song
    loader = EqloudLoader(filename=filepath, sampleRate=44100)
    audio = loader()
    return audio


def _extract_pitch_vector(audio):
    pitch_values, _pitch_confidence = pitch_extractor(audio)
    return pitch_values


def load_all_songs_pitch_vectors():
    '''
    Time:
    real    2m54,734s
    user    2m22,878s
    sys     0m15,550s
    '''
    pitch_vectors = []
    songs_paths = _read_songs_names()
    for song_path in songs_paths[:10]:
        filepath = '{}{}'.format(PATH_TO_DATASET, song_path)
        audio = _load_song(filepath)
        pitch_vector = _extract_pitch_vector(audio)
        pitch_vectors.append(pitch_vector)

    return pitch_vectors
