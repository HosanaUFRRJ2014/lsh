from argparse import ArgumentParser

from constants import DEFAULT_NUMBER_OF_PERMUTATIONS
from lsh import (
    calculate_jaccard_similarities,
    create_index,
    search
)


def tests():
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

    documents = []

    for i in range(5):
        document_path = 'tests/test_dataset/documents/doc{:02d}.txt'.format(i + 1)
        with open(document_path, 'r') as document_file:
            doc = document_file.read()
            documents.append(doc)

    for i in range(7):
        print('\nReading query {:02d}'.format(i + 1))
        query_path = 'tests/test_dataset/queries/query{:02d}.txt'.format(i + 1)
        with open(query_path, 'r') as query_file:
            query_content = query_file.read()
            inverted_index = create_index(
                documents_list=documents,
                num_permutations=num_permutations
            )
            similar_documents_count = search(
                query=query_content,
                inverted_index=inverted_index,
                num_permutations=num_permutations
            )
            jaccard_similarities = calculate_jaccard_similarities(
                query_content, similar_documents_count, documents
            )

        for doc_index, sim_percent in jaccard_similarities:
            print('document {} is {}% similar'.format(doc_index, sim_percent))
        print('jaccard_similarities: ', jaccard_similarities)
    return


if __name__ == '__main__':
    tests()
