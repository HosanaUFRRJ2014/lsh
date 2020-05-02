# -*-coding:utf8;-*-
import sys
import numpy as np
from argparse import ArgumentParser
from constants import (
    DEFAULT_NUMBER_OF_PERMUTATIONS,
    SERIALIZE_PITCH_VECTORS,
    CREATE_INDEX,
    PLSH_INDEX,
    NLSH_INDEX,
    INDEX_TYPES,
    SEARCH,
    SEARCH_ALL,
    SEARCH_METHODS,
    REQUIRE_INDEX_TYPE,
    METHODS,
    LINEAR_SCALING,
    RECURSIVE_ALIGNMENT,
    KTRA,
    MATCHING_ALGORITHMS,
    SHOW_TOP_X
)

from json_manipulator import (
    load_structure,
    serialize_pitch_contour_segmentations,
    deserialize_songs_pitch_contour_segmentations,
    deserialize_queries_pitch_contour_segmentations
)
from lsh import (
    apply_matching_algorithm,
    create_index,
    search,
    calculate_mean_reciprocal_rank
)
from loader import (
    load_song_pitch_contour_segmentation,
    load_expected_results
)

from messages import (
    log_invalid_index_type,
    log_invalid_matching_algorithm_error,
    log_invalid_method_error,
    log_no_dumped_files_error
)
from utils import (
    is_create_index_or_search_method,
    unzip_pitch_contours
)


def print_results(matching_algorithm, results, show_top_x):
    print('Results found by ', matching_algorithm)
    for query_name, result in results.items():
        print('Query: ', query_name)
        print('Results:')
        bounded_result = result[:show_top_x]
        for position, r in enumerate(bounded_result, start=1):
            print('\t{:03}. {}'.format(position, r))


def process_args():
    '''
    Processes program args.
    Returns a tuple containing the processed program args.
    '''
    parser = ArgumentParser()
    help_msg = "".join([
        "Number of permutations LSH will perform.",
        " Defaults to {}.".format(
            DEFAULT_NUMBER_OF_PERMUTATIONS
        )
    ])
    parser.add_argument(
        "method",
        type=str,
        help="Options: {}".format(
            ', '.join(METHODS)
        )
    )
    parser.add_argument(
        "--index_types",
        "-i",
        nargs='+',
        type=str,
        help="Options: {}. Required if \"method\" is any of \"{}\". Inform one or more.".format(
            ', '.join(INDEX_TYPES),
            ', '.join(REQUIRE_INDEX_TYPE)
        ),
        required=is_create_index_or_search_method(sys.argv)
    )
    parser.add_argument(
        "--song_filename",
        "-f",
        type=str,
        help="Filename for a song to search. NOTE: Loads song pitches.",
        default=''
    )
    parser.add_argument(
        "--number_of_permutations",
        "-np",
        type=int,
        help=help_msg,
        default=DEFAULT_NUMBER_OF_PERMUTATIONS
    )
    parser.add_argument(
        "--show_top",
        "-top",
        type=int,
        help="Shows top X results. Defaults to {}.".format(
            SHOW_TOP_X
        ),
        default=SHOW_TOP_X
    )
    parser.add_argument(
        "--matching_algorithm",
        "-ma",
        type=str,
        help="It's expected to be informed alongside {} methods. ".format(
            SEARCH_METHODS
        ) +
        "Options: {}. Defaults to {}".format(
            ', '.join(MATCHING_ALGORITHMS),
            LINEAR_SCALING
        ),
        default=LINEAR_SCALING
    )
    parser.add_argument(
        "--use_ls",
        type=bool,
        help="If {} and {} will include {}. Defaults to False.".format(
            RECURSIVE_ALIGNMENT, KTRA, LINEAR_SCALING
        ),
        default=False
    )
    args = parser.parse_args()
    num_permutations = args.number_of_permutations
    method_name = args.method
    index_types = args.index_types if args.index_types else []
    song_filename = args.song_filename
    matching_algorithm = args.matching_algorithm
    use_ls = args.use_ls
    show_top_x = args.show_top

    # Validate args
    invalid_method = method_name not in METHODS
    if invalid_method:
        log_invalid_method_error(method_name)
        exit(1)
    invalid_matching_algorithm = matching_algorithm not in MATCHING_ALGORITHMS
    if invalid_matching_algorithm:
        log_invalid_matching_algorithm_error(matching_algorithm)
        exit(1)

    invalid_index_type = len(
        set(INDEX_TYPES).union(set(index_types))
    ) != len(INDEX_TYPES)
    if invalid_index_type:
        log_invalid_index_type(index_types)
        exit(1)

    return method_name, \
        index_types, \
        song_filename, \
        num_permutations, \
        matching_algorithm, \
        use_ls, \
        show_top_x


def main():
    method_name,  \
        index_types,  \
        song_filename,  \
        num_permutations,  \
        matching_algorithm,  \
        use_ls,  \
        show_top_x = process_args()

    load_pitch_contour_segmentations = {
        SERIALIZE_PITCH_VECTORS: serialize_pitch_contour_segmentations,
        CREATE_INDEX: deserialize_songs_pitch_contour_segmentations,
        SEARCH_ALL: deserialize_queries_pitch_contour_segmentations,
        SEARCH: load_song_pitch_contour_segmentation
    }

    # Loading pitch vectors from audios
    # FIXME: Poor code
    if method_name == SEARCH:
        pitch_contour_segmentations = load_pitch_contour_segmentations[method_name](song_filename)
    else:
        pitch_contour_segmentations = load_pitch_contour_segmentations[method_name]()

    if method_name != SERIALIZE_PITCH_VECTORS:
        pitch_vectors = unzip_pitch_contours(pitch_contour_segmentations)

    if method_name == CREATE_INDEX:
        # Creating index(es)
        for index_type in index_types:
            create_index(
                pitch_contour_segmentations, index_type, num_permutations
            )
    elif method_name in SEARCH_METHODS:
        # Loading pitch vectors from json files
        # TODO: ADD NLSH INDEX here
        song_pitch_contour_segmentations = load_pitch_contour_segmentations[PLSH_INDEX]()
        song_pitch_vectors = unzip_pitch_contours(
            song_pitch_contour_segmentations
        )

        # Searching songs
        inverted_index = None
        audio_mapping = None
        try:
            inverted_index, audio_mapping, original_positions_mapping = (
                load_structure(structure_name=index_name)
                for index_name in [
                    'inverted_index',
                    'audio_mapping',
                    'original_positions_mapping'
                ]
            )
        except Exception as e:
            log_no_dumped_files_error(e)
            exit(1)

        candidates_indexes, candidates = search(
            pitch_contour_segmentations,
            inverted_index=inverted_index,
            songs_list=song_pitch_vectors,
            num_permutations=num_permutations
        )

        results = apply_matching_algorithm(
            choosed_algorithm=matching_algorithm,
            query=pitch_vectors,
            candidates_indexes=candidates_indexes,
            candidates=candidates,
            original_positions_mapping=original_positions_mapping,
            use_ls=use_ls
        )
        print_results(matching_algorithm, results, show_top_x)
        mrr = calculate_mean_reciprocal_rank(results, show_top_x)

        print('Mean Reciprocal Ranking (MRR): ', mrr)


if __name__ == '__main__':
    main()
