# -*-coding:utf8;-*-
import logging
from argparse import ArgumentParser
from constants import (
    DEFAULT_NUMBER_OF_PERMUTATIONS,
    CREATE_INDEX,
    SEARCH,
    VALID_METHODS
)

from json_manipulator import load_index
from lsh import (
    calculate_jaccard_similarities,
    create_index,
    search
)
from loader import (
    load_all_songs_pitch_vectors,
    load_all_queries_pitch_vectors
)


def _is_valid_method(method):
    return method in VALID_METHODS


def execute_method(method_name, num_permutations):
    result = None
    load_pitch_vectors = {
        CREATE_INDEX: load_all_songs_pitch_vectors,
        SEARCH: load_all_queries_pitch_vectors,
        # TODO: Search an especific song method (Informing one or more query names)
    }

    # Loading pitch vectors from audios
    pitch_vectors = load_pitch_vectors[method_name]()

    if method_name == CREATE_INDEX:
        # Creating index
        result = create_index(pitch_vectors, num_permutations)
    elif method_name == SEARCH:
        # Loading pitch vectors from audios
        # TODO: save it already loaded on a file?
        song_pitch_vectors = load_pitch_vectors[CREATE_INDEX]()
        # Searching songs
        inverted_index = None
        audio_mapping = None
        try:
            inverted_index = load_index(filename='inverted_index.json')
            audio_mapping = load_index(filename='audio_mapping.json')
        except Exception as e:
            logging.error(e)
            logging.error(
                'ERROR: Couldn\'t load inverted index or audio mapping. ' +
                'Are they dumped as files at all? ' +
                'Use method \'{}\' to generate the inverted index and audio mapping first.'.format(
                    CREATE_INDEX
                )
            )
            exit(1)
        similar_audios_count = search(pitch_vectors, inverted_index, song_pitch_vectors, num_permutations)

    return result


def main():
    parser = ArgumentParser()
    help_msg = "".join([
        "(Optional) Number of permutations LSH will perform.",
        " Defaults to {}.".format(
            DEFAULT_NUMBER_OF_PERMUTATIONS
        )
    ])
    parser.add_argument(
        "method",
        type=str,
        help="Options: 'create_index' or 'search'"
    )
    parser.add_argument(
        "--number_of_permutations",
        "-np",
        type=int,
        help=help_msg,
        default=DEFAULT_NUMBER_OF_PERMUTATIONS
    )
    args = parser.parse_args()
    num_permutations = args.number_of_permutations
    method_name = args.method

    is_invalid_method = not _is_valid_method(method_name)
    if is_invalid_method:
        print(
            "Method '{}' is invalid. Valid methods are: {}".format(
                method_name,
                VALID_METHODS
            )
        )
        exit(1)

    execute_method(method_name, num_permutations)

    # documents_list = ['Python é bom', 'Python é legal', 'Hosana Gomes', 'Eu quero dormir']
    # query = 'fefe'
    # inverted_index = create_index(documents_list, num_permutations)



    # Loading songs
    # songs_pitch_vectors = load_all_songs_pitch_vectors()

    # Creating inverted index
    # inverted_index = create_index(songs_pitch_vectors, num_permutations)

    # Seaching
    # queries_pitch_vectors = load_all_queries_pitch_vectors()

    '''
    jaccard_similarities = lsh(
        documents_list=documents_list,
        query=query,
        num_permutations=num_permutations
    )

    for doc_index, sim_percent in jaccard_similarities:
        print('document {} is {}% similar'.format(doc_index, sim_percent))
    print('jaccard_similarities: ', jaccard_similarities)
    '''


if __name__ == '__main__':
    main()
