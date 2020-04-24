# -*-coding:utf8;-*-
from json import JSONEncoder, dump, load
import numpy as np
from constants import JSON_PATH
from loader import (
    load_all_songs_pitch_vectors,
    load_all_queries_pitch_vectors
)
from messages import log_no_serialized_pitch_vectors_error


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def dump_structure(structure, structure_name, cls=NumpyArrayEncoder):
    '''
    Dumps Numpy ndarray or Python objects. Defaults to numpy objects.
    '''
    filename = '{}/{}.json'.format(JSON_PATH, structure_name)
    with open(filename, 'w') as json_file:
        dump(structure, json_file, cls=cls)


def load_structure(structure_name):
    '''
    Loads Numpy ndarray objects.
    '''
    filename = '{}/{}.json'.format(JSON_PATH, structure_name)
    with open(filename, 'r') as json_file:
        loaded = load(json_file)
        loaded = np.asarray(loaded)

    return loaded


def serialize_pitch_vectors():
    loader_functions_and_names = [
        (load_all_songs_pitch_vectors, 'songs_pitch_vectors'),
        (load_all_queries_pitch_vectors, 'queries_pitch_vectors')
    ]

    for loader_function, structure_name in loader_functions_and_names:
        pitch_vectors = loader_function()
        dump_structure(structure=pitch_vectors, structure_name=structure_name)


def _deserialize_pitch_vectors(structure_name):
    pitch_vectors = []
    try:
        pitch_vectors = load_structure(structure_name=structure_name)
    except FileNotFoundError:
        log_no_serialized_pitch_vectors_error(structure_name)
        exit(1)
    return pitch_vectors


def deserialize_songs_pitch_vectors():
    structure_name = 'songs_pitch_vectors'
    pitch_vectors = _deserialize_pitch_vectors(structure_name=structure_name)
    return pitch_vectors


def deserialize_queries_pitch_vectors():
    structure_name = 'queries_pitch_vectors'
    pitch_vectors = _deserialize_pitch_vectors(structure_name=structure_name)
    return pitch_vectors
