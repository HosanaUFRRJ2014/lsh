# -*-coding:utf8;-*-
from json import JSONEncoder, dump, load
import numpy as np


# TODO: Implement ComplexEncoder, once I doubt numpy array will be serialized
# in an easy way!
# Scratch, but will be more complex, once there is inner numpy arrays
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def dump_index(index, index_name):
    # print('Inverted index in json_manipulator.py')
    # print(index)
    filename = '{}.json'.format(index_name)
    with open(filename, 'w') as json_file:
        dump(index, json_file, cls=NumpyArrayEncoder)


def load_index(filename):
    index = None
    with open(filename, 'r') as json_file:
        loaded_index = load(json_file)
        index = np.asarray(loaded_index)

    return index
