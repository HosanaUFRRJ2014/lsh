from argparse import ArgumentParser

from lsh import lsh


def tests():
    parser = ArgumentParser()
    parser.add_argument(
        "number_of_permutations",
        type=int,
        help="Number of permutations"
    )
    args = parser.parse_args()
    num_permutations = args.np

    documents = []

    for i in range(5):
        document_path = 'tests/test_dataset/documents/doc{:02d}.txt'.format(i+1)
        with open(document_path, 'r') as document_file:
            doc = document_file.read()
            documents.append(doc)

    for i in range(7):
        query_path = 'tests/test_dataset/queries/query{:02d}.txt'.format(i+1)
        with open(query_path, 'r') as query_file:
            query_content = query_file.read()
            lsh(documents, query_content, num_permutations)

    return


if __name__ == '__main__':
    tests()