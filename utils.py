# -*-coding:utf8;-*-
import numpy as np
from constants import (
    INDEX_TYPES,
    MATCHING_ALGORITHMS,
    METHODS,
    REQUIRE_INDEX_TYPE
)
from messages import (
    log_invalid_index_type,
    log_invalid_matching_algorithm_error,
    log_invalid_method_error
)


def is_create_index_or_search_method(args):
    '''
    Says if passed method is creation or search of any index
    '''
    is_index_method = any([
        index_type
        for index_type in args
        if index_type in REQUIRE_INDEX_TYPE
    ])

    return is_index_method


def percent(part, whole):
    '''
    Given a percent and a whole part, calculates its real value.
    Ex:
    percent(10, 1000) # Ten percent of a thousand
    > 100
    '''
    return float(whole) / 100 * float(part)


def print_confidence_measurements(confidence_measurements):
    '''
    Prints confidence measurements of all queries.
    '''
    print('*' * 80)
    for query_name, candidates_and_measures in confidence_measurements.items():
        print('Query: ', query_name)
        pluralize = '' if len(candidates_and_measures) == 1 else 's'
        print('Candidate{0} confidence measurement{0}:'.format(pluralize))
        for candidate_and_measure in candidates_and_measures:
            print('\t', candidate_and_measure)
    print('*' * 80)


def print_results(matching_algorithm, index_type, results, show_top_x):
    print('*' * 80)
    print(f'Results found by {matching_algorithm} in {index_type}')
    for query_name, result in results.items():
        print('Query: ', query_name)
        print('Results:')
        bounded_result = result[:show_top_x]
        for position, r in enumerate(bounded_result, start=1):
            print('\t{:03}. {}'.format(position, r))
    print('*' * 80)


def train_confidence(all_confidence_measurements, results_mapping):
    confidence_training_data = []
    for query_name, candidates_and_measures in all_confidence_measurements.items():
        correct_result = results_mapping[query_name]
        first_candidate_name, first_candidate_measure = candidates_and_measures[0]
        if first_candidate_name != correct_result:
            confidence_training_data.append(first_candidate_measure)

    threshold = max(confidence_training_data)
    threshold_filename = 'confidence_threshold.txt'
    print(
        f'Max confidence measure is: {threshold}.\n',
        f'Saving in file {threshold_filename}'
    )
    with open(threshold_filename, 'w') as file:
        file.write(str(threshold))

    print("WARN: Exiting program because 'train_confidence' is True")
    exit(0)


def unzip_pitch_contours(pitch_contour_segmentations):
    """
    Extracts audio path and pitch vector for application of matching algorithms.
    """
    pitch_vectors = []
    for pitch_contour_segmentation in pitch_contour_segmentations:
        audio_path, pitch_vector, onsets, durations = pitch_contour_segmentation
        pitch_vectors.append((audio_path, pitch_vector))

    return np.array(pitch_vectors)


def validate_program_args(**kwargs):
    """
    Validates the list of program args. If any of them is invalid, logs an
    error message and exists program.
    Arguments:
        kwargs {dict} -- Dict of program args
    """
    method_name = kwargs['method_name']
    matching_algorithm = kwargs['matching_algorithm']
    index_types = kwargs['index_types']

    invalid_method = method_name not in METHODS
    invalid_matching_algorithm = matching_algorithm not in MATCHING_ALGORITHMS
    invalid_index_type = len(
        set(INDEX_TYPES).union(set(index_types))
    ) != len(INDEX_TYPES)

    if invalid_method:
        log_invalid_method_error(method_name)
        exit(1)
    if invalid_matching_algorithm:
        log_invalid_matching_algorithm_error(matching_algorithm)
        exit(1)
    if invalid_index_type:
        log_invalid_index_type(index_types)
        exit(1)
