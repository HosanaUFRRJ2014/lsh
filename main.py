# -*-coding:utf8;-*-
from argparse import ArgumentParser
from constants import DEFAULT_NUMBER_OF_PERMUTATIONS
from lsh import lsh


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

    documents_list = ['fefe']
    query = 'fefe'

    
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
