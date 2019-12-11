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
    matrix_n = len(documents)
    inverted_index = lil_matrix(
        (matrix_m, matrix_n),
        dtype=np.ndarray
    )

    fingerprints = np.array(range(1, td_matrix.shape[0] + 1))
    for j in range(matrix_n):
        # dj é um vetor bidimensional aleatório
        # (para representar uma música que já extraímos o mínimo e o máximo).
        dj = []
        for i in range(permutation_count):
            dj_permutation = permutate(
                to_permute=td_matrix[:, j],
                shuffle_seed=permutation_count,
                fingerprints=fingerprints
            )
            print(dj_permutation)
            for l in range(SELECTION_FUNCTION_COUNT):
                first_index = _get_index(
                    permutation_number=i,
                    selecion_function_code=l
                )
                second_index = SELECTION_FUNCTIONS[l](dj_permutation)
                if inverted_index[first_index][0].size == 0:
                    pass
                    # inverted_index[first_index][0] = np.array([j])
                else:
                    inverted_index[first_index][second_index].add(j)

                print("\t \t %d ª funcao: (%s) -> indice_invertido[%d][%d].add(%d)"%(l+1,SELECTION_FUNCTIONS[l], _get_index(i, l),SELECTION_FUNCTIONS[l](dj_permutation),j))

    return inverted_index


def main():
    documents = [d1, d2, d3, d4]
    vocabulary = {}
    td_matrix = tokenize(documents, vocabulary)
    print(td_matrix)
    print(vocabulary)

    inverted_index = generate_inverted_index(
        documents, td_matrix, 4
    )

    for i in inverted_index:
        print(i)


if __name__ == '__main__':
    main()
