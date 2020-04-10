# -*-coding:utf8;-*-
from copy import copy
import numpy as np
from math import floor
from random import sample, randint
from scipy.sparse import lil_matrix
from argparse import ArgumentParser

from constants import (
    SELECTION_FUNCTIONS,
    SELECTION_FUNCTION_COUNT,
    JACCARD_SIMILARITY,
    LINEAR_SCALING
)

from json_manipulator import dump_index

# __all__ = ["lsh"]
__all__ = ["calculate_jaccard_similarities", "create_index", "search"]


# def get_document_chunks(document):
def get_audio_chunks(pitch_values):
    '''
    Split the pitch vector into vectors.
    '''
    EXTRACTING_INTERVAL = 2
    WINDOW_SHIFT = 15
    WINDOW_LENGTH = 60
    pitch_vectors = []
    window_start = 0
    number_of_windows = len(pitch_values) / (WINDOW_SHIFT)
    number_of_windows = floor(number_of_windows)
    # # positions_in_original_song = []
    for window in range(number_of_windows):
        window_end = window_start + WINDOW_LENGTH
        pitch_vector = pitch_values[window_start:window_end:EXTRACTING_INTERVAL]
        pitch_vectors.append(pitch_vector)
        window_start += WINDOW_SHIFT

    return pitch_vectors


def _get_index(permutation_number, selecion_function_code=0):
    '''
    Calculate index of inverted index matrix.
    '''
    return permutation_number * (SELECTION_FUNCTION_COUNT) + selecion_function_code


def _vocab_index(term, vocabulary):
    # TODO: Does it need to take all the pitch vector to be the key?
    # dumped_term = term
    # copied = set([int(p) for p in term])
    # copied = sorted(copied)

    # vector_size = int(len(copied) / 4)
    # vector_size = len(copied)
    dumped_term = ''.join(str(p) for p in term)
    # dumped_term = ''.join(str(int(p)) for p in term)

    if dumped_term not in vocabulary.keys():
        vocabulary[dumped_term] = len(vocabulary) + 1

    return vocabulary[dumped_term], dumped_term


def _audio_mapping_index(dumped_term, audio_map, filename):
    if dumped_term not in audio_map.keys():
        audio_map[dumped_term] = np.array([])
    audio_map[dumped_term] = np.union1d(audio_map[dumped_term], filename)
    return audio_map


def tokenize(audios):
    # TODO: Trocar pela indexação pitch-vizinhos??
    vocabulary = {}
    audio_map = {}
    td_matrix_temp = []
    audios_length = len(audios)
    for i in range(audios_length):
        di_terms = []
        filename, audio = audios[i]
        audio_chunks = get_audio_chunks(audio)
        for termj in audio_chunks:
            index, dumped_term = _vocab_index(termj, vocabulary)
            di_terms.append(index)
            audio_map = _audio_mapping_index(dumped_term, audio_map, filename)
        td_matrix_temp.append(di_terms)
        di_terms = None
    td_matrix = np.zeros([len(vocabulary), audios_length])

    for i in range(audios_length):
        # TODO: Verify if I can realy ignore empty ones. Why there are empties?
        if td_matrix_temp[i]:
            td_matrix[np.array(td_matrix_temp[i]) - 1, i] = 1

    del td_matrix_temp
    td_matrix_temp = None
    return td_matrix, audio_map


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
    num_columns = td_matrix.shape[0]  # + 1
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

                # TODO: Verify if I can ignore empties.
                # Shouldn't I remove zero pitches at the read moment, like ref [16]
                # of the base article says?
                if non_zero_indexes[0].size > 0:
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
    similar_audios_count = np.zeros((inverted_index.shape[1] + 1, ), dtype=int)
    num_lines = permutation_count * SELECTION_FUNCTION_COUNT
    num_columns = query_td_matrix.shape[0]  # + 1

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

                # TODO: Verify if I can ignore empties.
                # Shouldn't I remove zero pitches at the reading moment, like ref [16]
                # of the base article says?
                if non_zero_indexes[0].size > 0:
                    second_index = int(
                        SELECTION_FUNCTIONS[l](dj_permutation[non_zero_indexes])
                    ) - 1

                    try:
                        retrieved_pitch_vector = inverted_index[first_index][second_index]
                        if not isinstance(retrieved_pitch_vector, int):
                            similar_audios_count[retrieved_pitch_vector] += 1
                        # print("retrieved pitch vector for fingerprint %d : "%(second_index), retrieved_pitch_vector)
                    except IndexError as e:
                        continue
    return similar_audios_count

    # non_zero_indexes = np.nonzero(suspicious)
    # suspicious = np.unique(suspicious[non_zero_indexes])
    # return suspicious


def calculate_jaccard_similarity(query_audio, similar_audio):
    '''
    Jaccard Similarity algorithm in steps:
    1. Get the shared members between both sets, i.e. intersection.
    2. Get the members in both sets (shared and un-shared, i.e. union).
    3. Divide the number of shared members found in (1) by the total number of
       members found in (2).
    4. Multiply the found number in (3) by 100.
    '''
    query_chunks = get_audio_chunks(query_audio.tolist())
    similar_audio_chunks = get_audio_chunks(similar_audio.tolist())

    intersection = np.intersect1d(similar_audio_chunks, query_chunks)
    union = np.union1d(similar_audio_chunks, query_chunks)

    jaccard_similarity = 0
    if union.size > 0:
        jaccard_similarity = (intersection.size / union.size) * 100

    return jaccard_similarity


def calculate_jaccard_similarities(query_audios, similar_audios_indexes, similar_audios, audio_mapping=None):
    '''
    Calculates jaccard similarity for all similar audios found in lsh search.

    :return: List of tuples. For each tuple, first position represents
    audio index number. The second position is the Jaccard Similarity
    with the query. This list of similarities is ordered from the most to
    the less similar.
    '''
    jaccards = {}
    for query_filename, query_audio in query_audios:
        jaccard_similarities = {}
        for audio_index, similarity_tuple in zip(similar_audios_indexes, similar_audios):
            similar_audio_filename, similar_audio = similarity_tuple
            jaccard_similarities[similar_audio_filename] = calculate_jaccard_similarity(
                query_audio, similar_audio
            )
        jaccard_similarities = sorted(
            jaccard_similarities.items(),
            key=lambda sim: sim[1],
            reverse=True
        )
        jaccards[query_filename] = jaccard_similarities

    return jaccards


def rescale_audio(query_audio, similar_audio):
    additional_length = len(similar_audio) - len(query_audio)
    rescaled_audio = query_audio + [0.0 for i in range(additional_length)]
    return rescaled_audio


def substract_vectors(similar_audio, rescaled_query_audio):
    result = np.absolute(
        np.subtract(similar_audio, rescaled_query_audio)
    )

    return result


def calculate_linear_scaling(query_audios, similar_audios_indexes, similar_audios):
    distances = {}
    for query_audio_name, query_audio in query_audios:
        linear_scaling = dict()
        for audio_index, similar_audio_tuple in zip(similar_audios_indexes, similar_audios):
            # TODO: passar a usar o segundo valor da tupla nome/vetor
            similar_audio_filename, similar_audio = similar_audio_tuple
            rescaled_query_audio = rescale_audio(
                query_audio.tolist(),
                similar_audio.tolist()
            )
            result = substract_vectors(similar_audio, rescaled_query_audio)
            linear_scaling[similar_audio_filename] = sum(result)
        linear_scaling = sorted(
            linear_scaling.items(),
            key=lambda res: res[1],
            reverse=True
        )
        distances[query_audio_name] = linear_scaling

    return distances


def apply_matching_algorithm(
    choosed_algorithm, query, similar_audios_indexes, similar_audios
):
    matching_algorithms = {
        JACCARD_SIMILARITY: calculate_jaccard_similarities,
        LINEAR_SCALING: calculate_linear_scaling
    }

    results = matching_algorithms[choosed_algorithm](
        query,
        similar_audios_indexes,
        similar_audios
    )
    return results


def create_index(audios_list, num_permutations):
    # Indexing
    audios = np.array(audios_list)
    td_matrix, audio_mapping = tokenize(audios)
    inverted_index = generate_inverted_index(td_matrix, num_permutations)

    # Serialize auxiliar index into a file
    # print()
    dump_index(audio_mapping, index_name='audio_mapping')

    # Serialize index into a file
    dump_index(inverted_index, index_name='inverted_index')

    # return inverted_index


def search(query, inverted_index, songs_list, num_permutations):
    songs = np.array(songs_list)
    query_td_matrix, query_audio_mapping = tokenize(query)
    similar_audios_count = search_inverted_index(
        query_td_matrix, inverted_index, num_permutations
    )
    similar_audios_indexes = (np.nonzero(similar_audios_count)[0] - 1)
    similar_songs = songs[similar_audios_indexes]

    return similar_audios_indexes, similar_songs
