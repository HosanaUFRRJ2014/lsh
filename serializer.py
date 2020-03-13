# -*-coding:utf8;-*-
import json
from json import JSONEncoder
import numpy


# TODO: Implement ComplexEncoder, once I doubt numpy array will be serialized
# in an easy way!
# Scratch, but will be more complex, once there is inner numpy arrays
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def dump_index(inverted_index):
    print('Inverted index in serializer.py')
    print(inverted_index)
    with open('inverted_index.json', 'w') as json_file:
        json.dump(inverted_index, json_file, cls=NumpyArrayEncoder)
