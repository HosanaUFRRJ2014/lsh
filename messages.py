import logging
from varname import nameof
from constants import (
    AUDIO_TYPES,
    CREATE_INDEX,
    FILE_TYPE_EXTENSIONS,
    INDEX_TYPES,
    SERIALIZE_PITCH_VECTORS,
    SERIALIZE_OPTIONS,
    MATCHING_ALGORITHMS,
    MANHATTAN_DISTANCE,
    METHODS
)


def log_bare_exception_error(err):
    message = 'A unexpected problem occured.'
    logging.error(err)
    logging.error(message)


def log_could_not_calculate_mrr_warning(query_name):
    message = " ".join([
        f"Could not calculate Mean Reciprocal Ranking for \"{query_name}\",",
        "because correct result was not found among candidates."
    ])
    logging.warn(message)


def log_forgotten_step_warn(error, audio_type):
    message = " ".join([
        f"{error}.\n Hint: \nYou may have forgotten to perform {audio_type}"
        "tfidf calculation step before."
    ])
    logging.warn(message)


def log_impossible_serialize_option_error():
    message = " ".join([
        "You must inform at least one of serialization options",
        SERIALIZE_OPTIONS
    ])
    logging.error(message)


def log_invalid_audio_type_error(audio_type):
    message = ' '.join([
        f"'{audio_type}' is not a valid audio type.",
        f"Options are {AUDIO_TYPES}."
    ])
    logging.error(message)


def log_invalid_index_type_error(index_types):
    message = ' '.join([
        f"'{index_types}' is(are) not (a) valid index(es) type(s).",
        f"Options are {INDEX_TYPES}."
    ])
    logging.error(message)


def log_invalid_matching_algorithm_error(matching_algorithm):
    if matching_algorithm == MANHATTAN_DISTANCE:
        reason_message = f"{MANHATTAN_DISTANCE} is invalid in LSH search context. Use this only for TF-IDF."
    else:
        reason_message = f"Value '{matching_algorithm}' for matching algorithm is invalid."
    message = ' '.join([
        reason_message,
        f"Valid algorithms are: {MATCHING_ALGORITHMS}."
    ])
    logging.error(message)


def log_invalid_method_error(method_name):
    message = f"Method '{method_name}' is invalid. Valid methods are: {METHODS}."
    logging.error(message)


def log_no_confidence_measurement_found_error():
    message = ' '.join([
        'Confidence measurement file was not found.',
        'Train confidence measurement first. Example usage:',
        "'python main.py search_all --train_confidence true'"
    ])
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


def log_seconds_not_found_warn(song_name, note_name, pitch):
    message = " ".join([
        f"{song_name}: Second was not found for note {note_name}",
        f"(pitch, {pitch}). Assuming value as None.",
        "You will need to filter out these none values if you try to use NLSH",
        "indexing and searching. Remember to fix onsets too."
    ])
    logging.warn(message)


def log_unsupported_file_extension_error(audio_path, extension):
    message = " ".join([
        f" '{audio_path}' has an unsupported file extension '{extension}'.", 
        f"Valid extensions are {FILE_TYPE_EXTENSIONS}"
    ])
    logging.error(message)


def log_useless_arg_warn(arg, purpose):
    arg_name = nameof(arg)
    message = f"The argument '{arg_name}' is useless for {purpose}"
    logging.warn(message)


def log_wrong_confidence_measurement_error(value):
    message = ' '.join([
        f'Value {value} of type {type(value)} is not a valid',
        'confidence measurement value.',
        'Expected type <float> .'
    ])
    logging.error(message)

