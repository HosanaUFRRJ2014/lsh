# -*-coding:utf8;-*-
from json import JSONEncoder, dump, load
from math import ceil
import numpy as np
from constants import JSON_PATH
from loader import (
    get_songs_count,
    get_queries_count,
    load_all_songs_pitch_contour_segmentations,
    load_all_queries_pitch_contour_segmentations
)
from messages import log_no_serialized_pitch_contour_segmentations_error


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def dump_structure(structure, structure_name, cls=NumpyArrayEncoder):
    '''
    Dumps Numpy ndarray or Python objects. Defaults to numpy objects.
    '''
    filename = f'{JSON_PATH}/{structure_name}.json'
    with open(filename, 'w') as json_file:
        dump(structure, json_file, cls=cls)


def load_structure(structure_name):
    '''
    Loads Numpy ndarray objects.
    '''
    filename = f'{JSON_PATH}/{structure_name}.json'
    with open(filename, 'r') as json_file:
        loaded = load(json_file)
        loaded = np.asarray(loaded)

    return loaded


def serialize_pitch_contour_segmentations():
    '''
    Serializes onsets, durations and pitch vectors of the songs and queries.
    '''
    counters_loaders_and_names = [
        # (
        #     get_songs_count,
        #     load_all_songs_pitch_contour_segmentations,
        #     'songs_pitch_contour_segmentations'
        # ),
        (
            get_queries_count,
            load_all_queries_pitch_contour_segmentations,
            'queries_pitch_contour_segmentations'
        )
    ]

    for get_count, loader_function, structure_name in counters_loaders_and_names:
        audios_count = 100 # get_count()

        size = ceil(audios_count / multiprocessing.cpu_count())  # BATCH_SIZE
        serialized_files = []
        batches_count = ceil(audios_count / size)
        start = 0
        end = size
        for batch_index in range(batches_count):
            batch_id = batch_index + 1
            print(f'Batch {batch_id} of {batches_count}')
            pitch_contour_segmentations = loader_function(start=start, end=end)
            batch_filename = f'{structure_name}_{batch_id}'
            dump_structure(
                structure=pitch_contour_segmentations,
                structure_name=batch_filename
            )
            serialized_files.append(batch_filename)
            start = end
            end += size

        # Saves serialized filenames in a file,
        # in order to process them in deserialization fase
        file_of_filenames = f'{structure_name}_filenames'
        dump_structure(
            structure=serialized_files,
            structure_name=file_of_filenames
        )


def _deserialize_pitch_contour_segmentations(structure_name):
    pitch_contour_segmentations = []
    try:
        pitch_contour_segmentations = load_structure(
            structure_name=structure_name
        )
    except FileNotFoundError:
        log_no_serialized_pitch_contour_segmentations_error(structure_name)
        exit(1)
    return pitch_contour_segmentations


def deserialize_songs_pitch_contour_segmentations():
    structure_name = 'songs_pitch_contour_segmentations'
    pitch_contour_segmentations = _deserialize_pitch_contour_segmentations(
        structure_name=structure_name
    )
    return pitch_contour_segmentations


def deserialize_queries_pitch_contour_segmentations():
    structure_name = 'queries_pitch_contour_segmentations'
    pitch_contour_segmentations = _deserialize_pitch_contour_segmentations(
        structure_name=structure_name
    )
    return pitch_contour_segmentations
