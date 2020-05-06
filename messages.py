import logging
from constants import (
    CREATE_INDEX,
    INDEX_TYPES,
    SERIALIZE_PITCH_VECTORS,
    MATCHING_ALGORITHMS,
    METHODS
)


def log_invalid_index_type(index_types):
    message = ' '.join([
        f"'{index_types}' is(are) not (a) valid(s) index(es) type(s).",
        f"Options are {INDEX_TYPES}."
    ])
    logging.error(message)


def log_invalid_matching_algorithm_error(matching_algorithm):
    message = ' '.join([
        f"Matching Algorithm '{matching_algorithm}' is invalid.",
        f"Valid algorithms are: {MATCHING_ALGORITHMS}."
    ])
    logging.error(message)


def log_invalid_method_error(method_name):
    message = "Method '{method_name}' is invalid. Valid methods are: {METHODS}."
    logging.error(message)


def log_no_dumped_files_error(original_error):
    message = ' '.join([
        "ERROR: Couldn't load inverted index or audio mapping.",
        "Are they dumped as files at all?",
        f"Use method '{CREATE_INDEX}'",
        "to generate the inverted index and audio mapping first."
    ])
    logging.error(original_error)
    logging.error(message)


def log_no_serialized_pitch_contour_segmentations_error(structure_name):
    message = ' '.join([
        f"ERROR: Couldn't load {structure_name}.",
        f"Use method '{SERIALIZE_PITCH_VECTORS}',",
        "to serialize pitch vectors first."
    ])
    logging.error(message)
