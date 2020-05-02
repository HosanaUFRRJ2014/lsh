from essentia.standard import (
    EqloudLoader,
    MusicExtractor,
    PitchContourSegmentation,
    PredominantPitchMelodia
)
import numpy as np
from constants import (
    FILENAMES_OF_SONGS,
    PATH_TO_DATASET,
    WAV_SONGS_PATH,
    FILENAMES_OF_QUERIES,
    QUERIES_PATH,
    EXPECTED_RESULTS
)

Extractor = MusicExtractor()

__all__ = ["load_all_songs_pitch_contour_segmentations", "load_all_queries_pitch_contour_segmentations"]


def _format_path(name, audio_path=None):
    if audio_path:
        formatted = '{}/{}/{}'.format(
            PATH_TO_DATASET, audio_path, name.rstrip('\n')
        )
    else:
        formatted = '{}/{}'.format(
            PATH_TO_DATASET, name.rstrip('\n')
        )
    return formatted


def _read_dataset_names(path, audio_path):
    filenames_file = open(path, 'r')

    paths = [
        _format_path(name, audio_path=audio_path)
        for name in filenames_file.readlines()
    ]
    filenames_file.close()
    return paths


def _read_expected_results(filename):
    results_file = open(filename, 'r')
    results_list = results_file.readlines()
    results_mapping = {}
    for result in results_list:
        query_path, song_name = result.split('\t')
        query = _format_path(query_path)
        song_name = song_name.replace('\n', '.wav')
        song = _format_path(song_name, audio_path=WAV_SONGS_PATH)
        results_mapping[query] = song

    return results_mapping


def _load_audio(filepath):
    # TODO: Calculate some features in other function, not all of them
    # features, features_frames = Extractor.compute(filepath)
    # features_keys = features.descriptorNames()

    # Loads the song
    loader = EqloudLoader(filename=filepath, sampleRate=44100)
    audio = loader()
    return audio


def _extract_pitch_values(audio):
    pitch_extractor = PredominantPitchMelodia(frameSize=2048, hopSize=128)
    pitch_values, _pitch_confidence = pitch_extractor(audio)
    return pitch_values


def _load_audio_pitch_contour_segmentation(audio_path):
    '''
    Returns the audio path, the pitch vector, the onsets and durations of
    each pitch.
    '''
    audio = _load_audio(audio_path)
    pitch_values = _extract_pitch_values(audio)
    # Removes zeros from the beginning and the end of the audio
    pitch_values = np.trim_zeros(pitch_values)

    contour_segmentator = PitchContourSegmentation()
    onsets, durations, _midipitches = contour_segmentator(pitch_values, audio)
    return audio_path, pitch_values, onsets, durations


def _load_all_audio_pitch_contour_segmentations(filenames_file, path):
    pitch_contour_segmentations = []
    audios_paths = _read_dataset_names(filenames_file, path)
    for audio_path in audios_paths[:100]:
        print('path: ', audio_path)
        pitch_contour_segmentations.append(
            _load_audio_pitch_contour_segmentation(audio_path)
        )

    return pitch_contour_segmentations


def load_song_pitch_contour_segmentation(audio_path):
    returned_tuple = _load_audio_pitch_contour_segmentation(audio_path)
    return np.array([returned_tuple])


def load_all_songs_pitch_contour_segmentations():
    return _load_all_audio_pitch_contour_segmentations(FILENAMES_OF_SONGS, WAV_SONGS_PATH)


def load_all_queries_pitch_contour_segmentations():
    return _load_all_audio_pitch_contour_segmentations(FILENAMES_OF_QUERIES, QUERIES_PATH)


def load_expected_results():
    '''
    Maps each query into its expected result.
    '''
    results_mapping = _read_expected_results(EXPECTED_RESULTS)
    return results_mapping
