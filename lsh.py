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
        td_matrix[np.array(td_matrix_temp[i]) - 1, i] = 1

        # print(documents[i])
        # print(td_matrix)
        # print('='*10)

    del td_matrix_temp
    td_matrix_temp = None
    return td_matrix


def get_fingerprint(vocabulary):
    return np.array(list(vocabulary.values()))


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


def generate_inverted_index(documents, vocabulary):
    inverted = [] # np.array([])
    test_matrix = []
    for d_index, document in enumerate(documents):
        terms = document.split(" ")
        fingerprint = np.array([
            vocabulary.get(term)
            for term in terms
        # Outra forma de recuperar a fingerprint: pegar os índices da td_matrix
        ])

        permutations = generate_permutations(
            fingerprint, number_of_permutations=len(terms)
        )

        # TODO: Optimize it! Try to no use enumerate (looks like expensive)
        line = np.zeros([len(permutations) * len(SELECTION_FUNCTIONS)])
        size = len(permutations) * len(SELECTION_FUNCTIONS)
        test_line = ['' for i in range(size)]
        for i, permutation in enumerate(permutations, start=0): # Start in 1?
            for j, function in enumerate(SELECTION_FUNCTIONS):
                index = i * NUMBER_OF_SELECTION_FUNCTIONS + j
                line[index] = SELECTION_FUNCTIONS[j](permutation)
                func_name = 'min' if j == 0 else 'max'
                test_line[index] = '{}(p{}(d{}))'.format(func_name, i, d_index)
        
        # inverted = np.append(inverted, [line])
        inverted.append(line)
        test_matrix.append(test_line)

    print('Matriz de teste:')
    for l in test_matrix:
        print(l)
    
    return inverted




def main():
    documents = [d1, d2, d3, d4]
    vocabulary = {}
    td_matrix = tokenize(documents, vocabulary)
    print(td_matrix)
    print(vocabulary)


    inverted_index = generate_inverted_index(documents, vocabulary)

    for i in inverted_index:
        print(i)

if __name__ == '__main__':
    main()
