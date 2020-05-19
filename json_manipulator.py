# -*-coding:utf8;-*-
from json import JSONEncoder, dump, load
from math import ceil
import multiprocessing
import numpy as np
from scipy.sparse import isspmatrix
from constants import (
    BATCH_SIZE,
    JSON_PATH
)
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


def load_structure(structure_name, as_numpy=True):
    '''
    Loads Numpy ndarray objects.
    '''
    filename = f'{JSON_PATH}/{structure_name}.json'
    with open(filename, 'r') as json_file:
        loaded = load(json_file)

        if as_numpy:
            loaded = np.asarray(loaded)

    return loaded


def _serialize(args):
    # print(f'Batch {batch_id} of {batches_count}')
    loader_function = args[0]
    structure_name = args[1]
    batch_id = args[2]
    start = args[3]
    end = args[4]
    pitch_contour_segmentations = loader_function(start=start, end=end)
    batch_filename = f'{structure_name}_{batch_id}'
    dump_structure(
        structure=pitch_contour_segmentations,
        structure_name=batch_filename
    )

    print(
        '%s says that %s%s is %s' % (
            multiprocessing.current_process().name,
            _serialize.__name__, batch_id, batch_filename
        )
    )

    del pitch_contour_segmentations

    return batch_filename


def serialize_pitch_contour_segmentations():
    '''
    Serializes onsets, durations and pitch vectors of the songs and queries.
    '''
    counters_loaders_and_names = [
        (
            get_songs_count,
            load_all_songs_pitch_contour_segmentations,
            'songs_pitch_contour_segmentations'
        ),
        (
            get_queries_count,
            load_all_queries_pitch_contour_segmentations,
            'queries_pitch_contour_segmentations'
        )
    ]

    for get_count, loader_function, structure_name in counters_loaders_and_names:
        audios_count = get_count()

        chunk_size = BATCH_SIZE
        batches_count = ceil(audios_count / chunk_size)
        num_processes = multiprocessing.cpu_count()
        tasks = []
        start = 0
        end = chunk_size
        for batch_id in range(1, batches_count + 1):
            tasks.append(
                (loader_function, structure_name, batch_id, start, end)
            )
            start = end
            end += chunk_size

        with multiprocessing.Pool(num_processes) as pool:
            results = [
                pool.apply_async(_serialize, (task, ))
                for task in tasks
            ]

            serialized_files = [
                result.get()
                for result in results
            ]

        # Saves serialized filenames in a file,
        # in order to process them in deserialization fase
        file_of_filenames = f'{structure_name}_filenames'
        dump_structure(
            structure=serialized_files,
            structure_name=file_of_filenames
        )


def _deserialize_pitch_contour_segmentations(file_of_filenames, num_audios=None):
    pitch_contour_segmentations = []
    try:
        list_of_files = load_structure(
            structure_name=file_of_filenames
        )
    except FileNotFoundError:
        log_no_serialized_pitch_contour_segmentations_error(file_of_filenames)
        exit(1)

    for filename in list_of_files:
        batch_pitch_contours = load_structure(structure_name=filename)
        if num_audios and len(batch_pitch_contours) > num_audios:
            batch_pitch_contours = batch_pitch_contours[:num_audios]
            pitch_contour_segmentations = batch_pitch_contours
            break
        pitch_contour_segmentations.extend(batch_pitch_contours)

    return pitch_contour_segmentations


def deserialize_songs_pitch_contour_segmentations(num_audios=None):
    file_of_filenames = 'songs_pitch_contour_segmentations_filenames'
    pitch_contour_segmentations = _deserialize_pitch_contour_segmentations(
        file_of_filenames=file_of_filenames,
        num_audios=num_audios
    )
    return pitch_contour_segmentations


def deserialize_queries_pitch_contour_segmentations(num_audios=None):
    file_of_filenames = 'queries_pitch_contour_segmentations_filenames'
    pitch_contour_segmentations = _deserialize_pitch_contour_segmentations(
        file_of_filenames=file_of_filenames,
        num_audios=num_audios
    )

    return pitch_contour_segmentations
