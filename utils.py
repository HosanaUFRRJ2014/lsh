# -*-coding:utf8;-*-
import numpy as np
from constants import REQUIRE_INDEX_TYPE


def unzip_pitch_contours(pitch_contour_segmentations):
    """
    Extracts audio path and pitch vector for application of matching algorithms.
    """
    pitch_vectors = []
    for pitch_contour_segmentation in pitch_contour_segmentations:
        audio_path, pitch_vector, onsets, durations = pitch_contour_segmentation
        pitch_vectors.append((audio_path, pitch_vector))

    return np.array(pitch_vectors)


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


def print_results(matching_algorithm, index_type, results, show_top_x):
    print('Results found by {} in {}'.format(matching_algorithm, index_type))
    for query_name, result in results.items():
        print('Query: ', query_name)
        print('Results:')
        bounded_result = result[:show_top_x]
        for position, r in enumerate(bounded_result, start=1):
            print('\t{:03}. {}'.format(position, r))
