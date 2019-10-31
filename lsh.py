# -*-coding:utf8;-*-
# qpy:3
# qpy:console
import random
from copy import copy
import numpy as np

d1 = "Hosana Gomes"
d2 = "python é bom 1"
d3 = "Estou com sono"
d4 = "Vivo com bastante fome"
vocabulary = {}


SELECTION_FUNCTIONS = [
    min,
    max
]
NUMBER_OF_SELECTION_FUNCTIONS = len(SELECTION_FUNCTIONS)

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
        td_matrix[np.array(td_matrix_temp[i])-1, i] = 1

        # print(documents[i])
        # print(td_matrix)
        # print('='*10)

    del td_matrix_temp
    td_matrix_temp = None
    return td_matrix


def generate_fingerprint(vocabulary):
    return np.array(list(vocabulary.values()))
    # fingerprint = [
    #     len(v) for v in vocabulary
    #]
    # return fingerprint


def permutate(fingerprint, shuffle_seed):
    # import ipdb; ipdb.set_trace()
    shuffled_list = copy(fingerprint)
    np.random.seed(seed=shuffle_seed)
    np.random.shuffle(shuffled_list)
    return shuffled_list


def generate_permutations(fingerprint, number_of_permutations=2):
    # TODO: Improve it! Looks poor
    permutations = [
        permutate(fingerprint, i)
        for i in range(number_of_permutations + 1)
    ]
    return permutations


def selection_function(function_index, v, permutation):
    # FIXME: A função de seleção é aplicada sobre um documento, não sobre um dict inteiro
    keys_array = np.array(list(v.keys()))
    functions = {
        1: keys_array[permutation[0] - 1]  # First minimal value
    }
    return functions[function_index]


def generate_inverted_index(documents, permutations)


def main():
    documents = [d1, d2, d3, d4]
    td_matrix = tokenize(documents, vocabulary)
    print(td_matrix)
    print(vocabulary)


    # exit()


if __name__ == '__main__':
    main()
