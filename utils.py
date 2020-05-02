# -*-coding:utf8;-*-
from constants import REQUIRE_INDEX_TYPE


def unzip_pitch_contours(pitch_contour_segmentations):
    """
    Extracts audio path and pitch vector for application of matching algorithms.
    """
    pitch_vectors = []
    for pitch_contour_segmentation in pitch_contour_segmentations:
        audio_path, pitch_vector, onsets, durations = pitch_contour_segmentation
        pitch_vectors.append((audio_path, pitch_vector))

    return pitch_vectors


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
