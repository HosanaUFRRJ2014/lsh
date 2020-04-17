# -*-coding:utf8;-*-
from sys import float_info
from copy import copy
import numpy as np
from math import floor
from random import sample, randint
from scipy.sparse import lil_matrix
from scipy.ndimage.interpolation import shift
from argparse import ArgumentParser
from essentia.standard import Mean

from constants import (
    SELECTION_FUNCTIONS,
    SELECTION_FUNCTION_COUNT,
    JACCARD_SIMILARITY,
    LINEAR_SCALING,
    BALS,
    BALS_SHIFT_SIZE,
    RECURSIVE_ALIGNMENT,
    MAX_RA_DEPTH,
    KTRA,
    MAX_KTRA_DEPTH,
    INITIAL_KTRA_K_VALUE
)

from json_manipulator import dump_structure, NumpyArrayEncoder


__all__ = ["create_index", "search"]


MAX_FLOAT = float_info.max
mean = Mean()


def get_audio_chunks(pitch_values, include_original_positions=False):
    '''
    Split the pitch vector into vectors.
    '''
    original_positions = []
    EXTRACTING_INTERVAL = 2
    WINDOW_SHIFT = 15
    WINDOW_LENGTH = 60
    pitch_vectors = []
    window_start = 0

    # Removes zerors from the start and the beginning of the audio
    pitch_values = np.trim_zeros(pitch_values)

    number_of_windows = len(pitch_values) / (WINDOW_SHIFT)
    number_of_windows = floor(number_of_windows)
    # # positions_in_original_song = []
    for window in range(number_of_windows):
        window_end = window_start + WINDOW_LENGTH
        pitch_vector = pitch_values[window_start:window_end:EXTRACTING_INTERVAL]
        original_positions.append(window_start)
        pitch_vectors.append(pitch_vector)
        window_start += WINDOW_SHIFT

    if include_original_positions:
        return pitch_vectors, original_positions
    else:
        return pitch_vectors


def _get_index(permutation_number, selecion_function_code=0):
    '''
    Calculate index of inverted index matrix.
    '''
    return permutation_number * (SELECTION_FUNCTION_COUNT) + selecion_function_code


def _dump_term(term):
    return ''.join(str(p) for p in term)


def _vocab_index(term, vocabulary):
    # TODO: Does it need to take all the pitch vector to be the key?
    # dumped_term = term
    # copied = set([int(p) for p in term])
    # copied = sorted(copied)

    # vector_size = int(len(copied) / 4)
    # vector_size = len(copied)
    dumped_term = _dump_term(term)
    # dumped_term = ''.join(str(int(p)) for p in term)

    if dumped_term not in vocabulary.keys():
        vocabulary[dumped_term] = len(vocabulary) + 1

    return vocabulary[dumped_term], dumped_term


def _audio_mapping_index(dumped_term, audio_map, filename):
    if dumped_term not in audio_map.keys():
        audio_map[dumped_term] = np.array([])
    audio_map[dumped_term] = np.union1d(audio_map[dumped_term], filename)
    return audio_map


def _original_position_index(dumped_term, orig_pos_map, original_pos, filename):
    '''
    For a given filename, stores its dumped terms and their original positions.
    '''
    if filename not in orig_pos_map.keys():
        orig_pos_map[filename] = np.array([])
    tuple_of_infos = (dumped_term, original_pos)
    orig_pos_map[filename] = np.union1d(orig_pos_map[filename], tuple_of_infos)

    return orig_pos_map


def tokenize(audios):
    # TODO: Trocar pela indexação pitch-vizinhos??
    vocabulary = {}
    audio_map = {}
    orig_pos_map = {}
    td_matrix_temp = []
    audios_length = len(audios)
    for i in range(audios_length):
        di_terms = []
        filename, audio = audios[i]
        audio_chunks, original_positions = get_audio_chunks(
            audio,
            include_original_positions=True
        )
        for audio_chunk_index, termj in enumerate(audio_chunks):
            index, dumped_term = _vocab_index(termj, vocabulary)
            di_terms.append(index)
            audio_map = _audio_mapping_index(dumped_term, audio_map, filename)
            orig_pos_map = _original_position_index(
                dumped_term,
                orig_pos_map,
                audio_chunk_index,
                filename
            )
        td_matrix_temp.append(di_terms)
        di_terms = None
    td_matrix = np.zeros([len(vocabulary), audios_length])

    for i in range(audios_length):
        # TODO: Verify if I can really ignore empty ones. Why there are empties?
        if td_matrix_temp[i]:
            td_matrix[np.array(td_matrix_temp[i]) - 1, i] = 1

    del td_matrix_temp
    td_matrix_temp = None
    return td_matrix, audio_map, orig_pos_map


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
    candidates_count = np.zeros((inverted_index.shape[1] + 1, ), dtype=int)
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
                            candidates_count[retrieved_pitch_vector] += 1
                        # print("retrieved pitch vector for fingerprint %d : "%(second_index), retrieved_pitch_vector)
                    except IndexError as e:
                        continue
    return candidates_count


def calculate_jaccard_similarity(query_audio, candidate, **kwargs):
    '''
    Jaccard Similarity algorithm in steps:
    1. Get the shared members between both sets, i.e. intersection.
    2. Get the members in both sets (shared and un-shared, i.e. union).
    3. Divide the number of shared members found in (1) by the total number of
       members found in (2).
    4. Multiply the found number in (3) by 100.
    '''
    query_chunks = get_audio_chunks(query_audio.tolist())
    candidate_chunks = get_audio_chunks(candidate.tolist())

    intersection = np.intersect1d(candidate_chunks, query_chunks)
    union = np.union1d(candidate_chunks, query_chunks)

    jaccard_similarity = 0
    if union.size > 0:
        jaccard_similarity = (intersection.size / union.size) * 100

    return jaccard_similarity


def rescale_audio(query_audio, candidate):
    additional_length = candidate.size - query_audio.size

    if additional_length > 0:
        rescaled_audio = np.append(query_audio, np.zeros(additional_length))
    elif additional_length < 0:
        rescaled_audio = query_audio[:candidate.size]
    else:
        # No need for rescaling
        rescaled_audio = query_audio
    return rescaled_audio


def calculate_manhattan_distance(rescaled_query_audio, candidate):
    result = np.absolute(
        np.subtract(candidate, rescaled_query_audio)
    )
    return sum(result)


def calculate_linear_scaling(query_audio, candidate, **kwargs):
    rescaled_query_audio = rescale_audio(
        query_audio,
        candidate
    )
    distance = calculate_manhattan_distance(
        rescaled_query_audio,
        candidate
    )
    if distance == 0.0:
        # Ignoring zero distance. (It's likely a noise)
        distance = MAX_FLOAT
    return distance


def get_candidate_neighbourhood(**kwargs):
    # FIXME: Looks a little bit (or a lot) wrong
    candidate = kwargs.get('candidate')
    candidate_name = kwargs.get('candidate_name')
    original_positions_mapping = kwargs.get('original_positions_mapping')
    neighbours = []
    # pitch_position_and_vectors = original_positions_mapping.get(candidate_name)
    # print('get_candidate_neighbourhood NOT IMPLEMENTED YET!!!!')

    # audio_chunks = get_audio_chunks(candidate)
    # for chunk in audio_chunks:
    #     dumped_term = _dump_term(chunk)
    #     corresponding = list(
    #         filter(
    #             lambda: position_and_vector: (position_and_vector[1] == dumped_term),
    #             pitch_position_and_vectors
    #         )
    #     )[0]
    #     position, _vector = corresponding
    # position = filter( pitch_vectors)

    # Apply shift, shorthen, lengthen operations
    # Left moved vector
    left_moved_vector = shift(candidate, BALS_SHIFT_SIZE)
    # Right moved vector
    right_moved_vector = shift(candidate, -BALS_SHIFT_SIZE)
    # TODO: Left shortened vector
    # TODO: Right shortened vector
    # Left lenghtened vector
    left_lenghtened = shift(candidate, BALS_SHIFT_SIZE, mode='nearest')
    # Right lenghtened vector
    right_lenghtened = shift(candidate, -BALS_SHIFT_SIZE, mode='nearest')

    # Tem algo que parece não estar certo... É para aplicar as operações sobre
    # cada trecho do áudio candidato ou sobre ele como um todo? Ou ainda,
    # sobre um único fragmento candidato?
    neighbours = [
        left_moved_vector,
        right_moved_vector,
        left_lenghtened,
        right_lenghtened
    ]

    return neighbours


def calculate_bals(query_audio, candidate, **kwargs):
    '''
    Explore candidates neighbourhood
       - For each candidate, lenghten or shorten it.
    For each neighbor, measure LS distance.
    Retain the fragment with the shortest distance.
    '''
    # FIXME: It's rescaling the same query several times
    candidate_distance = calculate_linear_scaling(query_audio, candidate)

    kwargs['candidate'] = candidate

    neighbours = get_candidate_neighbourhood(**kwargs)
    # Starts with the similar audio
    nearest_neighbour_distance = candidate_distance
    for neighbour in neighbours:
        distance = calculate_linear_scaling(query_audio, neighbour)
        if distance < nearest_neighbour_distance:
            nearest_neighbour_distance = distance

    return nearest_neighbour_distance


def percent(part, whole):
    '''
    Given a percent and a whole part, calculates its real value.
    Ex:
    percent(10, 1000) # Ten percent of a thousand
    > 100
    '''
    return float(whole) / 100 * float(part)


def recursive_align(query_audio, candidate, **kwargs):
    min_distance = calculate_linear_scaling(
        query_audio=query_audio,
        candidate=candidate
    )

    depth = kwargs.get('depth')

    if query_audio.size == 0 or candidate.size == 0:
        raise Exception('size zero detected!!!')

    if depth < MAX_RA_DEPTH:
        query_size = query_audio.size
        candidate_size = candidate.size
        query_portion_size = int((query_size / 2) + 1)
        # portion_percents = [10, 20, 30, 40, 50, 60, 70, 80, 90] # Too slow
        portion_percents = [40, 50, 60]
        for portion_percent in portion_percents:
            size = int(
                percent(portion_percent, candidate_size) + 1
            )
            complement_size = candidate_size + 1 - size
            left_query_portion = query_audio[:query_portion_size]
            right_query_portion = query_audio[query_portion_size:]
            left_similar_portion = candidate[:size]
            right_similar_portion = candidate[complement_size:]

            left_distance = recursive_align(
                left_query_portion,
                left_similar_portion,
                depth=depth + 1
            )

            right_distance = recursive_align(
                right_query_portion,
                right_similar_portion,
                depth=depth + 1
            )

            min_distance = min([left_distance, right_distance, min_distance])

    return min_distance


def mean_substract(pitch_vector):
    return pitch_vector - mean.compute(pitch_vector)


def calculate_ktra(query_audio, candidate, **kwargs):
    depth = kwargs.get('depth')

    if depth == 0:
        query_audio = mean_substract(query_audio)
        candidate = mean_substract(candidate)

    k = kwargs.get('k')

    d_minus = recursive_align(query_audio - k, candidate, depth=0)
    d_zero = recursive_align(query_audio, candidate, depth=0)
    d_plus = recursive_align(query_audio + k, candidate, depth=0)

    min_distance = min([d_minus, d_zero, d_plus])
    if depth < MAX_KTRA_DEPTH:
        # FIXME: Needs to treat iqual distances?
        if d_minus == min_distance:
            query_audio = query_audio - k
        elif d_plus == min_distance:
            query_audio = query_audio + k

        min_distance = calculate_ktra(
            query_audio,
            candidate,
            k=k / 2,
            depth=depth + 1
        )

    return min_distance


def apply_matching_algorithm(
    choosed_algorithm, query, candidates_indexes, candidates, original_positions_mapping
):
    matching_algorithms = {
        JACCARD_SIMILARITY: calculate_jaccard_similarity,
        LINEAR_SCALING: calculate_linear_scaling,
        BALS: calculate_bals,
        RECURSIVE_ALIGNMENT: recursive_align,
        KTRA: calculate_ktra
    }

    all_queries_distances = {}
    for query_audio_name, query_audio in query:
        query_audio = np.array(query_audio)
        query_audio = np.trim_zeros(query_audio)
        query_distance = dict()
        for audio_index, candidate_tuple in zip(candidates_indexes, candidates):
            candidate_filename, candidate = candidate_tuple
            candidate = np.array(candidate)
            candidate = np.trim_zeros(candidate)
            ##
            distance_or_similarity = matching_algorithms[choosed_algorithm](
                query_audio,
                candidate,
                query_audio_name=query_audio_name,
                original_positions_mapping=original_positions_mapping,
                depth=0,  # For recursive alignment and ktra
                k=INITIAL_KTRA_K_VALUE  # For ktra
            )
            ##
            query_distance[candidate_filename] = distance_or_similarity

        reverse_order = False
        if choosed_algorithm == JACCARD_SIMILARITY:
            reverse_order = True

        query_distance = sorted(
            query_distance.items(),
            key=lambda res: res[1],
            reverse=reverse_order
        )
        all_queries_distances[query_audio_name] = query_distance

        # print('Stopping earlier for debugging purposes')
        # break

    return all_queries_distances


def create_index(audios_list, num_permutations):
    # Indexing
    td_matrix, audio_mapping, original_positions_mapping = tokenize(audios_list)
    inverted_index = generate_inverted_index(td_matrix, num_permutations)

    # Serializing indexes
    indexes_and_indexes_names = [
        (inverted_index, 'inverted_index'),
        (audio_mapping, 'audio_mapping'),
        (original_positions_mapping, 'original_positions_mapping')
    ]
    for index, index_name in indexes_and_indexes_names:
        dump_structure(
            index,
            structure_name=index_name
        )


def search(query, inverted_index, songs_list, num_permutations):
    query_td_matrix, _query_audio_mapping, _query_positions_mapping = tokenize(
        query
    )
    candidates_count = search_inverted_index(
        query_td_matrix, inverted_index, num_permutations
    )
    candidates_indexes = (np.nonzero(candidates_count)[0] - 1)
    similar_songs = songs_list[candidates_indexes]

    return candidates_indexes, similar_songs
