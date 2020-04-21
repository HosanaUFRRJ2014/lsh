import logging
from constants import (
    CREATE_INDEX,
    SERIALIZE_PITCH_VECTORS,
    MATCHING_ALGORITHMS,
    METHODS
)


def log_invalid_matching_algorithm_error(matching_algorithm):
    message = "Matching Algorithm '{}' is invalid. Valid algorithms are: {}".format(
        matching_algorithm,
        MATCHING_ALGORITHMS
    )
    logging.error(message)


def log_invalid_method_error(method_name):
    message = "Method '{}' is invalid. Valid methods are: {}".format(
        method_name,
        METHODS
    )
    logging.error(message)


def log_no_dumped_files_error(original_error):
    message = ''.join([
        "ERROR: Couldn't load inverted index or audio mapping. ",
        "Are they dumped as files at all? ",
        "Use method '{}' ".format(CREATE_INDEX),
        "to generate the inverted index and audio mapping first."
    ])
    logging.error(original_error)
    logging.error(message)


def log_no_serialized_pitch_vectors_error(structure_name):
    message = ''.join([
        "ERROR: Couldn't load {}. ".format(structure_name),
        "Use method '{}', to serialize pitch vectors first.".format(
            SERIALIZE_PITCH_VECTORS
        )
    ])
    logging.error(message)
