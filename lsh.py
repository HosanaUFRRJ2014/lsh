# -*-coding:utf8;-*-
# qpy:3
# qpy:console
from random import sample, randint
from copy import copy
import numpy as np
from pprint import pprint
from scipy.sparse import lil_matrix

d1 = "Hosana Gomes"
d2 = "python é bom 1"
d3 = "Estou com sono"
d4 = "Vivo com bastante fome Hosana"
d5 = "Hosana Gomes python"


SELECTION_FUNCTIONS = [
    min,
    max
]
SELECTION_FUNCTION_COUNT = len(SELECTION_FUNCTIONS)


def _get_index(permutation_number, selecion_function_code=0):
    '''
    Calculate index of inverted index matrix.
    '''
    return permutation_number * (SELECTION_FUNCTION_COUNT) + selecion_function_code


def vocab_index(term, v):
    if term in v.keys():
        return v[term]
    else:
        v[term] = len(v) + 1
        return v[term]


def tokenize(documents, v):
    td_matrix_temp = []
    for i in range(len(documents)):
        di_terms = []
        for termj in documents[i].split(" "):
            di_terms.append(vocab_index(termj, v))
        td_matrix_temp.append(di_terms)
        di_terms = None
    td_matrix = np.zeros([len(v), len(documents)])
    # print(td_matrix.shape)
    for i in range(len(documents)):
        td_matrix[np.array(td_matrix_temp[i]) - 1, i] = 1

        # print(documents[i])
        # print(td_matrix)
        # print('='*10)

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


def generate_permutations(to_permute, number_of_permutations, fingerprints):
    # TODO: Improve it! Looks poor
    permutations = [
        permutate(to_permute, i, fingerprints)
        for i in range(number_of_permutations + 1)
    ]
    return permutations


def generate_inverted_index(
    documents, td_matrix, permutation_count
):
    matrix_m = permutation_count * (SELECTION_FUNCTION_COUNT + 1) + SELECTION_FUNCTION_COUNT
    matrix_n = td_matrix.shape[0] + 1
    inverted_index = np.zeros(
        (matrix_m, matrix_n),
        dtype=np.ndarray
    )

    fingerprints = np.array(range(1, matrix_n))
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
                X = np.nonzero(dj_permutation)
                second_index = int(SELECTION_FUNCTIONS[l](dj_permutation[X]))
                print("(%d, %d) on (%d, %d)"%(first_index, second_index, matrix_m, matrix_n),dj_permutation[X])
                try:
                    inverted_index[first_index][second_index][0]==0
                    inverted_index[first_index][second_index] = np.append(inverted_index[first_index][second_index], [j+1])
                except:
                    inverted_index[first_index][second_index] = np.array([j+1])
                print("\t \t %d ª funcao: (%s) -> indice_invertido[%d][%d].add(%d)"%(l+1,SELECTION_FUNCTIONS[l], first_index,second_index,j+1))
                print("@@",inverted_index,"@@")

    return inverted_index


def search_inverted_index(
    query_td_matrix, inverted_index, permutation_count
):
    matrix_m = permutation_count * (SELECTION_FUNCTION_COUNT + 1) + SELECTION_FUNCTION_COUNT
    matrix_n = query_td_matrix.shape[0] + 1
    inverted_index = np.zeros(
        (matrix_m, matrix_n),
        dtype=np.ndarray
    )

    fingerprints = np.array(range(1, matrix_n))
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
                X = np.nonzero(dj_permutation)
                second_index = int(SELECTION_FUNCTIONS[l](dj_permutation[X]))
                retrieved_documents = inverted_index[first_index][second_index]
                print("retrieved documents for fingerprint %d : "%(second_index), retrieved_documents)
    return inverted_index


def main():
    documents = [d1, d2, d3, d4, d5]
    vocabulary = {}
    td_matrix = tokenize(documents, vocabulary)
    print(td_matrix)
    print(vocabulary)

    inverted_index = generate_inverted_index(
        documents, td_matrix, 4
    )

    query = d5
    q_td_matrix = tokenize([query], vocabulary)
    search_inverted_index(q_td_matrix, inverted_index, 4)

   


if __name__ == '__main__':
    main()
