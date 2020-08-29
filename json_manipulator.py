# -*-coding:utf8;-*-
import os
from json import JSONEncoder, dump, load
from math import ceil
import multiprocessing
import numpy as np
import pandas as pd
from constants import (
    BATCH_SIZE,
    FILES_PATH,
    QUERY,
    SONG
)
from loader import (
    get_songs_count,
    get_queries_count,
    load_all_songs_pitch_contour_segmentations,
    load_all_queries_pitch_contour_segmentations
)
from messages import (
    log_invalid_audio_type_error,
    log_no_serialized_pitch_contour_segmentations_error
)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def dump_structure(
    structure, structure_name, cls=NumpyArrayEncoder,
    as_numpy=True, as_pandas=False, extension="json"
):
    '''
    Dumps Numpy ndarray , Pandas or Python objects. Defaults to numpy objects.
    '''

    filename = f'{FILES_PATH}/{structure_name}.{extension}'

    filepath = "/".join(
        filename.split("/")[:-1]
    )

    if not os.path.exists(filepath) and filepath != FILES_PATH:
        os.mkdir(filepath)

    if as_numpy:
        with open(filename, 'w') as json_file:
            dump(structure, json_file, cls=cls)
    elif as_pandas:
        pd.to_pickle(structure, filename)


def load_structure(
    structure_name, as_numpy=True, as_pandas=False, extension="json"
):
    '''
    Loads Numpy ndarray, Pandas or simple read objects.
    '''
    filename = f'{FILES_PATH}/{structure_name}.{extension}'

    if not as_pandas:
        with open(filename, 'r') as json_file:
            loaded = load(json_file)

            if as_numpy:
                loaded = np.asarray(loaded)
    else:
        loaded = pd.read_pickle(filename)

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

    return batch_filename, pitch_contour_segmentations


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
        tasks = []
        start = 0
        end = chunk_size
        for batch_id in range(1, batches_count + 1):
            tasks.append(
                (loader_function, structure_name, batch_id, start, end)
            )
            start = end
            end += chunk_size

        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(num_processes) as pool:
            results = [
                pool.apply_async(_serialize, (task, ))
                for task in tasks
            ]

            batches_filenames = []
            pitches_countour_segmentations = []
            for result in results:
                batch_filename, batch_pitch_countour_segmentations = result.get()
                batches_filenames.append(batch_filename)
                pitches_countour_segmentations.extend(
                    batch_pitch_countour_segmentations
                )

        # Saves serialized filenames in a file,
        # in order to process them in deserialization fase
        file_of_filenames = f'{structure_name}_filenames'
        dump_structure(
            structure=batches_filenames,
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


def deserialize_audios_pitch_contour_segmentations(audio_type, num_audios=None):
    deserialize = {
        SONG: deserialize_songs_pitch_contour_segmentations,
        QUERY: deserialize_queries_pitch_contour_segmentations
    }

    deserialized_data = {}
    try:
        deserialized_data = deserialize[audio_type](num_audios)
    except KeyError:
        log_invalid_audio_type_error(audio_type)
        exit(1)
    
    return deserialized_data
    