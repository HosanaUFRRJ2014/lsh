# -*-coding:utf8;-*-
from json import JSONEncoder, dump, load
import numpy as np
from constants import JSON_PATH
from loader import (
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


def serialize_pitch_contour_segmentations():
    '''
    Serializes onsets, durations and pitch vectors of the songs and queries.
    '''
    loader_functions_and_names = [
        (load_all_songs_pitch_contour_segmentations, 'songs_pitch_contour_segmentations'),
        (load_all_queries_pitch_contour_segmentations, 'queries_pitch_contour_segmentations')
    ]

    for loader_function, structure_name in loader_functions_and_names:
        pitch_contour_segmentations = loader_function()
        dump_structure(
            structure=pitch_contour_segmentations,
            structure_name=structure_name
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
