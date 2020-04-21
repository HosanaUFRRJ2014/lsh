# -*-coding:utf8;-*-
import numpy as np
from argparse import ArgumentParser
from constants import (
    DEFAULT_NUMBER_OF_PERMUTATIONS,
    SERIALIZE_PITCH_VECTORS,
    CREATE_INDEX,
    SEARCH,
    SEARCH_ALL,
    SEARCH_METHODS,
    METHODS,
    LINEAR_SCALING,
    MATCHING_ALGORITHMS,
    SHOW_TOP_X
)

from json_manipulator import (
    load_structure,
    serialize_pitch_vectors,
    deserialize_songs_pitch_vectors,
    deserialize_queries_pitch_vectors
)
from lsh import (
    apply_matching_algorithm,
    create_index,
    search
)
from loader import (
    load_song_pitch_vector
)

from messages import (
    log_invalid_matching_algorithm_error,
    log_invalid_method_error,
    log_no_dumped_files_error
)


def print_results(matching_algorithm, results, show_top_x):
    print('Results found by ', matching_algorithm)
    for result_name, result in results.items():
        print('Query: ', result_name)
        print('Results:')
        bounded_result = result[:show_top_x]
        for position, r in enumerate(bounded_result, start=1):
            print('\t{:03}. {}'.format(position, r))


def process_args():
    '''
    Processes program args.
    Returns a tuple containing the program args.
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
        "--song_filename",
        "-f",
        type=str,
        help="Filename for a song to search. NOTE: Loads song pitches",
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
    args = parser.parse_args()
    num_permutations = args.number_of_permutations
    method_name = args.method
    song_filename = args.song_filename
    matching_algorithm = args.matching_algorithm
    show_top_x = args.show_top

    invalid_method = method_name not in METHODS
    if invalid_method:
        log_invalid_method_error(method_name)
        exit(1)
    invalid_matching_algorithm = matching_algorithm not in MATCHING_ALGORITHMS
    if invalid_matching_algorithm:
        log_invalid_matching_algorithm_error(matching_algorithm)
        exit(1)


    return method_name, song_filename, num_permutations, matching_algorithm, show_top_x


def main():
    method_name, song_filename, num_permutations, matching_algorithm, show_top_x = process_args()

    load_pitch_vectors = {
        SERIALIZE_PITCH_VECTORS: serialize_pitch_vectors,
        CREATE_INDEX: deserialize_songs_pitch_vectors,
        SEARCH_ALL: deserialize_queries_pitch_vectors,
        SEARCH: load_song_pitch_vector
    }

    # Loading pitch vectors from audios
    # FIXME: Poor code
    if method_name == SEARCH:
        pitch_vectors = load_pitch_vectors[method_name](song_filename)
    else:
        pitch_vectors = load_pitch_vectors[method_name]()

    if method_name == CREATE_INDEX:
        # Creating index
        create_index(pitch_vectors, num_permutations)
    elif method_name in SEARCH_METHODS:
        # Loading pitch vectors from json files
        song_pitch_vectors = load_pitch_vectors[CREATE_INDEX]()
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
            query=pitch_vectors,
            inverted_index=inverted_index,
            songs_list=song_pitch_vectors,
            num_permutations=num_permutations
        )

        results = apply_matching_algorithm(
            choosed_algorithm=matching_algorithm,
            query=pitch_vectors,
            candidates_indexes=candidates_indexes,
            candidates=candidates,
            original_positions_mapping=original_positions_mapping
        )
        print_results(matching_algorithm, results, show_top_x)


if __name__ == '__main__':
    main()
