from essentia.standard import (
    EqloudLoader,
    MusicExtractor,
    PitchContourSegmentation,
    PredominantPitchMelodia
)
import music21
import numpy as np
from constants import (
    FILENAMES_OF_SONGS,
    PATH_TO_DATASET,
    MIDI_SONGS_PATH,
    FILENAMES_OF_QUERIES,
    QUERIES_PATH,
    EXPECTED_RESULTS,
    MIDI,
    WAVE
)

from messages import log_unsupported_file_extension_error

Extractor = MusicExtractor()

__all__ = ["load_all_songs_pitch_contour_segmentations", "load_all_queries_pitch_contour_segmentations"]


def get_file_extension(audio_path):
    return audio_path.split(".")[-1]


def _format_path(name, audio_path=None):
    stripped_name = name.rstrip('\n')
    if audio_path:
        formatted = f'{PATH_TO_DATASET}/{audio_path}/{stripped_name}'
    else:
        formatted = f'{PATH_TO_DATASET}/{stripped_name}'
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
        song_name = song_name.replace('\n', '.mid')
        song = _format_path(song_name, audio_path=MIDI_SONGS_PATH)
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

def _extract_song_pitch_contour_segmentation(audio_path):
    '''
    Returns the audio path, the pitch vector, the onsets and durations of
    each pitch.
    '''
    audio = music21.converter.parse(audio_path)

    pitches = []
    durations = []
    onsets = []
    accumulated_time = 0
    for element in list(audio.recurse()):
        if isinstance(element, music21.note.Note):
            pitch_space = element.pitch.ps
            duration = element.seconds
            pitches.append(pitch_space)
            durations.append(duration)
            # Trying to estimate note onset here. It may not be correct.
            onsets.append(accumulated_time)
            accumulated_time += duration

    return audio_path, pitches, onsets, durations


def _extract_query_pitch_contour_segmentation(audio_path):
    '''
    Returns the audio path, the pitch vector, the onsets and durations of
    each pitch.
    '''
    audio = _load_audio(audio_path)
    pitch_values = _extract_pitch_values(audio)
    # Removes zeros from the beginning and the end of the audio
    pitch_values = np.trim_zeros(pitch_values)

    contour_segmentator = PitchContourSegmentation()
    onsets, durations, midipitches = contour_segmentator(pitch_values, audio)
    return audio_path, midipitches, onsets, durations


def _load_all_audio_pitch_contour_segmentations(filenames_file, path, extraction_function, start, end=None):
    pitch_contour_segmentations = []
    audios_paths = _read_dataset_names(filenames_file, path)

    NOT_WORKING_AUDIOS = [
        # Audios which can't be loaded. The reason is unknown.
        '../uniformiza_dataset/queries/004043.wav',
        '../uniformiza_dataset/queries/004048.wav',
        '../uniformiza_dataset/queries/004050.wav',
        '../uniformiza_dataset/queries/004051.wav'
    ]
    for audio_path in audios_paths[start:end]:
        print('path: ', audio_path)
        if audio_path in NOT_WORKING_AUDIOS:
            print(f'{audio_path}  skipped')
            continue
        pitch_contour_segmentations.append(
            extraction_function(audio_path)
        )

    return pitch_contour_segmentations


def _get_audios_count(filenames_file, path):
    audios_paths = _read_dataset_names(filenames_file, path)

    return len(audios_paths)


def get_songs_count():
    return _get_audios_count(
        filenames_file=FILENAMES_OF_SONGS,
        path=MIDI_SONGS_PATH
    )


def get_queries_count():
    return _get_audios_count(
        filenames_file=FILENAMES_OF_QUERIES,
        path=QUERIES_PATH,
    )


def _load_query_pitch_contour_segmentation(audio_path):
    return _extract_query_pitch_contour_segmentation(audio_path)


def _load_song_pitch_contour_segmentation(audio_path):
    return _extract_song_pitch_contour_segmentation(audio_path)


def load_audio_pitch_contour_segmentation(audio_path):
    extension = get_file_extension(audio_path)

    loader_function = {
        MIDI: _load_song_pitch_contour_segmentation,
        WAVE: _load_query_pitch_contour_segmentation
    }

    try:
        returned_tuple = loader_function[extension](audio_path)
    except KeyError:
        log_unsupported_file_extension_error(audio_path, extension)
        exit(1)

    return np.array([returned_tuple])

def load_all_songs_pitch_contour_segmentations(start=0, end=None):
    audio_pitch_contour_segmentations = _load_all_audio_pitch_contour_segmentations(
        filenames_file=FILENAMES_OF_SONGS,
        path=MIDI_SONGS_PATH,
        extraction_function=_extract_song_pitch_contour_segmentation,
        start=start,
        end=end
    )

    return audio_pitch_contour_segmentations


def load_all_queries_pitch_contour_segmentations(start=0, end=None):
    return _load_all_audio_pitch_contour_segmentations(
        filenames_file=FILENAMES_OF_QUERIES,
        path=QUERIES_PATH,
        extraction_function=_extract_query_pitch_contour_segmentation,
        start=start,
        end=end
    )


def load_expected_results():
    '''
    Maps each query into its expected result.
    '''
    results_mapping = _read_expected_results(EXPECTED_RESULTS)
    return results_mapping
