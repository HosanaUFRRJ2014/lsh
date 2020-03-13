# -*-coding:utf8;-*-
from argparse import ArgumentParser
from constants import DEFAULT_NUMBER_OF_PERMUTATIONS
from lsh import (
    calculate_jaccard_similarities,
    create_index,
    search
)
from loader import load_all_songs_pitch_vectors


def main():
    parser = ArgumentParser()
    help_msg = "".join([
        "(Optional) Number of permutations LSH will perform.",
        " Defaults to {}.".format(
            DEFAULT_NUMBER_OF_PERMUTATIONS
        )
    ])
    parser.add_argument(
        "--number_of_permutations",
        "-np",
        type=int,
        help=help_msg,
        default=DEFAULT_NUMBER_OF_PERMUTATIONS
    )
    args = parser.parse_args()
    num_permutations = args.number_of_permutations

    # documents_list = ['Python é bom', 'Python é legal', 'Hosana Gomes', 'Eu quero dormir']
    # query = 'fefe'
    # inverted_index = create_index(documents_list, num_permutations)

    pitch_vectors = load_all_songs_pitch_vectors()
    inverted_index = create_index(pitch_vectors, num_permutations)

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
