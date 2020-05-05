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
    create_indexes,
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
    log_invalid_method_error
)
from utils import (
    is_create_index_or_search_method,
    print_results
)


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

    if method_name == CREATE_INDEX:
        # Creating index(es)
        pitch_contour_segmentations = load_pitch_contour_segmentations[method_name]()
        create_indexes(
            pitch_contour_segmentations, index_types, num_permutations
        )
    elif method_name in SEARCH_METHODS:
        # Loading query and song pitch vectors
        if method_name == SEARCH:
            query_pitch_contour_segmentations = load_pitch_contour_segmentations[method_name](song_filename)
        elif method_name == SEARCH_ALL:
            query_pitch_contour_segmentations = load_pitch_contour_segmentations[method_name]()

        song_pitch_contour_segmentations = load_pitch_contour_segmentations[CREATE_INDEX]()

        # Searching and applying matching algorithms
        results = search(
            query_pitch_contour_segmentations=query_pitch_contour_segmentations,
            song_pitch_contour_segmentations=song_pitch_contour_segmentations,
            index_types=index_types,
            matching_algorithm=matching_algorithm,
            use_ls=use_ls,
            num_permutations=num_permutations
        )

        # Results and MRR
        print_results(matching_algorithm, index_types, results, show_top_x)
        mrr = calculate_mean_reciprocal_rank(results, show_top_x)

        print('Mean Reciprocal Ranking (MRR): ', mrr)


if __name__ == '__main__':
    main()


# TODO: Renomear pitch_vectors, posto que estes contém o audio path e
# o vetor de pitches
