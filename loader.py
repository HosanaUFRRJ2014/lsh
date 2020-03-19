from essentia.standard import (
    MusicExtractor,
    EqloudLoader,
    PredominantPitchMelodia
)
from constants import (
    FILENAMES_OF_SONGS,
    PATH_TO_DATASET,
    WAV_SONGS_PATH,
    FILENAMES_OF_QUERIES,
    QUERIES_PATH
)

Extractor = MusicExtractor()

__all__ = ["load_all_songs_pitch_vectors", "load_all_queries_pitch_vectors"]


def _read_dataset_names(path, audio_path):
    filenames_file = open(path, 'r')
    paths = [
        '{}/{}/{}'.format(PATH_TO_DATASET, audio_path, name.rstrip('\n'))
        for name in filenames_file.readlines()
    ]
    filenames_file.close()
    return paths


def _load_audio(filepath):
    # TODO: Calculate some features in other function, not all of them
    # features, features_frames = Extractor.compute(filepath)
    # features_keys = features.descriptorNames()

    # Loads the song
    loader = EqloudLoader(filename=filepath, sampleRate=44100)
    audio = loader()
    return audio


def _extract_pitch_vector(audio):
    pitch_extractor = PredominantPitchMelodia(frameSize=2048, hopSize=128)
    pitch_values, _pitch_confidence = pitch_extractor(audio)
    return pitch_values


def _load_all_audio_pitch_vectors(filenames_file, path):
    pitch_vectors = []
    audios_paths = _read_dataset_names(filenames_file, path)
    for audio_path in audios_paths[:16]:
        print('path: ', audio_path)
        audio = _load_audio(audio_path)
        pitch_vector = _extract_pitch_vector(audio)
        pitch_vectors.append(pitch_vector)

    return pitch_vectors


def load_all_songs_pitch_vectors():
    return _load_all_audio_pitch_vectors(FILENAMES_OF_SONGS, WAV_SONGS_PATH)


def load_all_queries_pitch_vectors():
    return _load_all_audio_pitch_vectors(FILENAMES_OF_QUERIES, QUERIES_PATH)
