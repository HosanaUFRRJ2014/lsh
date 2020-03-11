# -*-coding:utf8;-*-
from copy import copy
import numpy as np
from random import sample, randint
from scipy.sparse import lil_matrix
from argparse import ArgumentParser

from constants import (
    SELECTION_FUNCTIONS,
    SELECTION_FUNCTION_COUNT
)

# __all__ = ["lsh"]
__all__ = ["calculate_jaccard_similarities", "create_index", "search"]


def get_document_chunks(document):
    '''
    Split the document in chunks.
    '''
    if isinstance(document, list):
        document = document[0]
    return document.split(' ')


def _get_index(permutation_number, selecion_function_code=0):
    '''
    Calculate index of inverted index matrix.
    '''
    return permutation_number * (SELECTION_FUNCTION_COUNT) + selecion_function_code


def _vocab_index(term, v):
    if term in v.keys():
        return v[term]
    else:
        v[term] = len(v) + 1
        return v[term]


def tokenize(documents, v):
    td_matrix_temp = []
    for i in range(len(documents)):
        di_terms = []
        document_chunks = get_document_chunks(documents[i])
        for termj in document_chunks:
            di_terms.append(_vocab_index(termj, v))
        td_matrix_temp.append(di_terms)
        di_terms = None
    td_matrix = np.zeros([len(v), len(documents)])

    for i in range(len(documents)):
        td_matrix[np.array(td_matrix_temp[i]) - 1, i] = 1

    del td_matrix_temp
    td_matrix_temp = None
    return td_matrix


def get_fingerprint(vocabulary):
    return np.array(list(vocabulary.values()))


def permutate(to_permute, shuffle_seed, fingerprints):
    shuffled_list = copy(fingerprints)
    np.random.seed(seed=shuffle_seed)
    np.random.shuffle(shuffled_list)
    resultant_fingerprints = to_permute * shuffled_list

    return resultant_fingerprints


def generate_inverted_index(td_matrix, permutation_count):
    # num_lines = permutation_count * (SELECTION_FUNCTION_COUNT + 1) + SELECTION_FUNCTION_COUNT
    num_lines = permutation_count * SELECTION_FUNCTION_COUNT  # + SELECTION_FUNCTION_COUNT
    num_columns = td_matrix.shape[0] #+ 1
    inverted_index = np.zeros(
        (num_lines, num_columns),
        dtype=np.ndarray
    )

    fingerprints = np.array(range(1, num_columns + 1))
    for j in range(td_matrix.shape[1]):
        for i in range(permutation_count):
            dj_permutation = permutate(
                to_permute=td_matrix[:, j],
                shuffle_seed=i,
                fingerprints=fingerprints
            )
            for l in range(SELECTION_FUNCTION_COUNT):
                first_index = _get_index(
                    permutation_number=i,
                    selecion_function_code=l
                )
                non_zero_indexes = np.nonzero(dj_permutation)
                second_index = int(
                    SELECTION_FUNCTIONS[l](dj_permutation[non_zero_indexes])
                ) - 1
                # print("(%d, %d) on (%d, %d)"%(first_index, second_index, num_lines, num_columns),dj_permutation[non_zero_indexes])
                if isinstance(inverted_index[first_index][second_index], np.ndarray):
                    inverted_index[first_index][second_index] = np.append(
                        inverted_index[first_index][second_index],
                        [j + 1]
                    )
                else:
                    inverted_index[first_index][second_index] = np.array([j + 1])
                # print("\t \t %d ª funcao: (%s) -> indice_invertido[%d][%d].add(%d)"%(l+1,SELECTION_FUNCTIONS[l].__name__, first_index,second_index,j+1))

    return inverted_index


def search_inverted_index(
    query_td_matrix, inverted_index, permutation_count
):
    # num_lines = permutation_count * (SELECTION_FUNCTION_COUNT + 1) + SELECTION_FUNCTION_COUNT
    similar_docs_count = np.zeros((inverted_index.shape[1] + 1, ), dtype=int)
    num_lines = permutation_count * SELECTION_FUNCTION_COUNT
    num_columns = query_td_matrix.shape[0] # + 1

    fingerprints = np.array(range(1, num_columns + 1))
    for j in range(query_td_matrix.shape[1]):
        for i in range(permutation_count):
            dj_permutation = permutate(
                to_permute=query_td_matrix[:, j],
                shuffle_seed=i,
                fingerprints=fingerprints
            )
            for l in range(SELECTION_FUNCTION_COUNT):
                first_index = _get_index(
                    permutation_number=i,
                    selecion_function_code=l
                )
                non_zero_indexes = np.nonzero(dj_permutation)
                second_index = int(
                    SELECTION_FUNCTIONS[l](dj_permutation[non_zero_indexes])
                ) - 1

                try:
                    retrieved_documents = inverted_index[first_index][second_index]
                    # print('retrieved_documents:', retrieved_documents)
                    if not isinstance(retrieved_documents, int):
                        similar_docs_count[retrieved_documents] += 1
                    # print("retrieved documents for fingerprint %d : "%(second_index), retrieved_documents)
                except IndexError as e:
                    continue
    return similar_docs_count

    non_zero_indexes = np.nonzero(suspicious)
    suspicious = np.unique(suspicious[non_zero_indexes])
    return suspicious


def calculate_jaccard_similarity(query_document, similar_document):
    '''
    Jaccard Similarity algorithm in steps:
    1. Get the shared members between both sets, i.e. intersection.
    2. Get the members in both sets (shared and un-shared, i.e. union).
    3. Divide the number of shared members found in (1) by the total number of
       members found in (2).
    4. Multiply the found number in (3) by 100.
    '''
    query_chunks = get_document_chunks(query_document)
    similar_document_chunks = get_document_chunks(similar_document)

    intersection = set(similar_document_chunks).intersection(query_chunks)
    union = set(similar_document_chunks + query_chunks)

    jaccard_similarity = len(intersection) / len(union) * 100

    return jaccard_similarity


def calculate_jaccard_similarities(query_document, similar_docs_count, documents_list):
    '''
    Calculates jaccard similarity for all similar documents found in lsh search.

    :return: List of tuples. For each tuple, first position represents
    document index number. The second position is the Jaccard Similarity
    with the query. This list of similarities is ordered from the most to
    the less similar.
    '''
    documents = np.array(documents_list)
    similar_docs_indexes = (np.nonzero(similar_docs_count)[0] - 1)
    similar_documents = documents[similar_docs_indexes]
    jaccard_similarities = {
        doc_index + 1: calculate_jaccard_similarity(query_document, document)
        for doc_index, document in zip(similar_docs_indexes, similar_documents)
    }
    jaccard_similarities = sorted(
        jaccard_similarities.items(),
        key=lambda sim: sim[1],
        reverse=True
    )
    return jaccard_similarities


def create_index(documents_list, num_permutations):
    # Indexing
    vocabulary = {}
    documents = np.array(documents_list)
    td_matrix = tokenize(documents, vocabulary)
    inverted_index = generate_inverted_index(td_matrix, num_permutations)

    # TODO: serialize inverted_index
    return inverted_index


def search(query, inverted_index, num_permutations):
    # Query
    vocabulary = {}
    if not isinstance(query, list):
        query = [query]

    query_td_matrix = tokenize(query, vocabulary)
    similar_docs_count = search_inverted_index(
        query_td_matrix, inverted_index, num_permutations
    )

    return similar_docs_count

"""
def lsh(documents_list, query, num_permutations):
    ""/"
    Perform LSH permutation-based algorithm in a list of documents,given a
    query.

    Arguments:
        documents_list {list} -- list of documents which will be indexed.
        query {str} -- A single query
        num_permutations {int} -- number of permutations LSH will make

    Returns:
        list -- jaccard similarities, calculated over the list of similar
        documents found in LSH search and the query.
    ""/"
    # Indexing
    vocabulary = {}
    documents = np.array(documents_list)
    td_matrix = tokenize(documents, vocabulary)
    inverted_index = generate_inverted_index(td_matrix, num_permutations)

    # Query
    if not isinstance(query, list):
        query = [query]

    query_td_matrix = tokenize(query, vocabulary)
    similar_docs_count = search_inverted_index(
        query_td_matrix, inverted_index, num_permutations
    )

    # Jaccard Similarity
    jaccard_similarities = calculate_jaccard_similarities(
        query, similar_docs_count, documents
    )

    return jaccard_similarities
"""