# -*-coding:utf8;-*-
from sys import float_info
from copy import copy
import numpy as np
from math import floor, ceil
from random import sample, randint
from scipy.ndimage import find_objects
from scipy.ndimage.interpolation import shift
from argparse import ArgumentParser
from essentia.standard import Mean

from constants import (
    PLSH_INDEX,
    NLSH_INDEX,
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

from json_manipulator import (
    dump_structure,
    load_structure,
    NumpyArrayEncoder
)
from loader import load_expected_results

from messages import (
    log_no_dumped_files_error
)

from utils import (
    percent,
    print_confidence_measurements,
    train_confidence,
    unzip_pitch_contours
)

__all__ = ["create_indexes", "search"]


MAX_FLOAT = float_info.max
mean = Mean()


def exec_plsh_pitch_extraction(pitch_contour_segmentation, include_original_positions=False):
    '''
    Splits the pitch vector into vectors, according to PLSH indexing.
    '''
    _, pitch_values, _, _ = pitch_contour_segmentation
    original_positions = []
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
        original_positions.append(window_start)
        pitch_vectors.append(pitch_vector)
        window_start += WINDOW_SHIFT

    if include_original_positions:
        return pitch_vectors, original_positions
    else:
        return pitch_vectors


def exec_nlsh_pitch_extraction(
    pitch_contour_segmentation, include_original_positions=False
):
    '''
    Split the pitch vector into vectors, according to NLSH indexing.
    If a note is longer than MAX_LENGHT_L, it's splited into one or more notes.
    '''
    _, pitch_values, onsets, durations = pitch_contour_segmentation
    PITCHES_PER_SECOND = 25
    WINDOW_SHIFT = 3   # in article: 1
    WINDOW_LENGTH = 10
    MAX_LENGHT_L = 10

    original_positions = []
    pitch_vectors = []
    window_start = 0
    calculate_note_shift = lambda duration: duration * PITCHES_PER_SECOND

    # Splits notes greater than MAX_LENGHT_L
    processed_pitch_values = []
    get_pitch_index = lambda onset: int(onset * PITCHES_PER_SECOND)
    for onset, duration in zip(onsets, durations):
        processed_onsets = []
        processed_durations = []
        processed_onsets.append(onset)
        if duration > MAX_LENGHT_L:
            splited_duration = duration / MAX_LENGHT_L
            number_of_splits = ceil(duration / MAX_LENGHT_L)
            for i in range(1, number_of_splits + 1):
                new_onset = onset + splited_duration * i
                processed_onsets.append(new_onset)
                processed_durations.append(splited_duration)
        else:
            processed_durations.append(duration)

        for p_onset, _p_duration in zip(processed_onsets, processed_durations):
            index = get_pitch_index(p_onset)
            processed_pitch_values.append(pitch_values[index])

    # Generate pitch vectors
    window_start = 0
    number_of_windows = len(pitch_values) / (WINDOW_SHIFT)
    number_of_windows = floor(number_of_windows)
    for pitch in range(number_of_windows):
        window_end = window_start + WINDOW_LENGTH
        pitch_vector = pitch_values[window_start:window_end]
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


def _dump_piece(piece):
    return ''.join(str(p) for p in piece)


def _vocab_index(piece, vocabulary):
    # TODO: Does it need to take all the pitch vector to be the key?
    dumped_piece = _dump_piece(piece)
    if dumped_piece not in vocabulary.keys():
        vocabulary[dumped_piece] = len(vocabulary) + 1

    return vocabulary[dumped_piece], dumped_piece


def _audio_mapping_index(dumped_piece, audio_map, filename):
    if dumped_piece not in audio_map.keys():
        audio_map[dumped_piece] = np.array([])
    audio_map[dumped_piece] = np.union1d(audio_map[dumped_piece], filename)
    return audio_map


def _original_position_index(dumped_piece, orig_pos_map, original_pos, filename):
    '''
    For a given filename, stores its dumped pieces and their original positions.
    '''
    if filename not in orig_pos_map.keys():
        orig_pos_map[filename] = np.array([])
    tuple_of_infos = (dumped_piece, original_pos)
    orig_pos_map[filename] = np.union1d(orig_pos_map[filename], tuple_of_infos)

    return orig_pos_map


def tokenize(pitch_contour_segmentations, index_type):
    vocabulary = {}
    audio_map = {}
    orig_pos_map = {}
    td_matrix_temp = []
    number_of_audios = len(pitch_contour_segmentations)
    exec_pitch_extraction = {
        PLSH_INDEX: exec_plsh_pitch_extraction,
        NLSH_INDEX: exec_nlsh_pitch_extraction
    }
    for i in range(number_of_audios):
        di_pieces = []
        filename, _audio, _onsets, _durations = pitch_contour_segmentations[i]

        audio_chunks, original_positions = exec_pitch_extraction[index_type](
            pitch_contour_segmentations[i],
            include_original_positions=True
        )

        for audio_chunk_index, piecej in enumerate(audio_chunks):
            index, dumped_piece = _vocab_index(piecej, vocabulary)
            di_pieces.append(index)
            audio_map = _audio_mapping_index(dumped_piece, audio_map, filename)
            orig_pos_map = _original_position_index(
                dumped_piece,
                orig_pos_map,
                audio_chunk_index,
                filename
            )
        td_matrix_temp.append(di_pieces)
        di_pieces = None
    td_matrix = np.zeros([len(vocabulary), number_of_audios])

    for i in range(number_of_audios):
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
    query_chunks = exec_plsh_pitch_extraction(query_audio.tolist())
    candidate_chunks = exec_plsh_pitch_extraction(candidate.tolist())

    intersection = np.intersect1d(candidate_chunks, query_chunks)
    union = np.union1d(candidate_chunks, query_chunks)

    jaccard_similarity = 0
    if union.size > 0:
        jaccard_similarity = (intersection.size / union.size) * 100

    return jaccard_similarity


def rescale_audio(query_audio):
    # ------------------[0.5, 0.75, 1.0, 1.25, 1.5]
    scaling_factors = ((1, 2), (3, 4), (1, 1), (5, 4), (3, 2))
    original_len = query_audio.size
    rescaled_audios = []
    for numerator, denominator in scaling_factors:
        rescaled_audio = np.array([])
        scaling_factor = numerator / denominator
        if scaling_factor == 1.0:
            rescaled_audio = np.copy(query_audio)
        else:
            # Shorten or lenghten audio
            numerator, denominator = scaling_factor.as_integer_ratio()
            for i in range(0, original_len, denominator):
                chunck = query_audio[i:i + denominator]
                if scaling_factor > 1.0:
                    # Lenghten audio
                    # Note: only works for 1.25 (5,4) and 1.5 (3,2)
                    repeated = chunck[-1]
                    chunck = np.append(chunck, repeated)
                rescaled_audio = np.append(rescaled_audio, chunck[:numerator])
        rescaled_audios.append(rescaled_audio)

    return rescaled_audios


def calculate_manhattan_distance(rescaled_audio, candidate):
    additional_length = candidate.size - rescaled_audio.size

    # Equalize size
    if additional_length > 0:
        rescaled_audio = np.append(rescaled_audio, np.zeros(additional_length))
    elif additional_length < 0:
        rescaled_audio = rescaled_audio[:candidate.size]

    # Calculate distance
    result = np.absolute(
        np.subtract(candidate, rescaled_audio)
    )
    return sum(result)


def calculate_linear_scaling(
    rescaled_query_audios, candidate, **kwargs
):
    '''
    Implemented as explained in "Query-By-Singing-and-Hummimg" from CHIAO-WEI
    LIN.
    '''
    include_zero_distance = kwargs.get('include_zero_distance')
    if not isinstance(rescaled_query_audios, list):
        rescaled_query_audios = [rescaled_query_audios]
    distances = []
    for rescaled_query_audio in rescaled_query_audios:
        distance = calculate_manhattan_distance(
            rescaled_query_audio,
            candidate
        )
        if distance > 0.0 or include_zero_distance:
            distances.append((distance, rescaled_query_audio))
    if not distance:
        # Ignoring zero distance. (It's likely a noise)
        min_distance = MAX_FLOAT, None
    else:
        min_distance, query = min(distances, key=lambda t: t[0])
    return min_distance, query


def get_candidate_neighbourhood(**kwargs):
    # FIXME: Looks a little bit (or a lot) wrong
    candidate = kwargs.get('candidate')
    candidate_name = kwargs.get('candidate_name')
    original_positions_mapping = kwargs.get('original_positions_mapping')
    neighbours = []
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


def calculate_bals(rescaled_query_audios, candidate, **kwargs):
    '''
    Explore candidates neighbourhood
       - For each candidate, lenghten or shorten it.
    For each neighbor, measure LS distance.
    Retain the fragment with the shortest distance.
    '''
    candidate_distance, _query = calculate_linear_scaling(
        rescaled_query_audios, candidate, include_zero_distance=True
    )

    kwargs['candidate'] = candidate

    neighbours = get_candidate_neighbourhood(**kwargs)
    # Starts with the similar audio
    nearest_neighbour_distance = candidate_distance
    for neighbour in neighbours:
        distance, _query = calculate_linear_scaling(
            rescaled_query_audios,
            neighbour,
            include_zero_distance=True
        )
        if distance < nearest_neighbour_distance:
            nearest_neighbour_distance = distance

    return nearest_neighbour_distance


def recursive_align(query_audio, candidate, **kwargs):
    # Compute the linear distance of the corresponding part
    min_distance, rescaled_query_audio = calculate_linear_scaling(
        query_audio,
        candidate=candidate,
        include_zero_distance=False
    )

    depth = kwargs.get('depth')

    if rescaled_query_audio.size == 0 or candidate.size == 0:
        raise Exception('size zero detected!!!')

    if depth < MAX_RA_DEPTH:
        query_size = rescaled_query_audio.size
        candidate_size = candidate.size
        query_portion_size = int((query_size / 2) + 1)
        # portion_percents = [10, 20, 30, 40, 50, 60, 70, 80, 90] # Too slow
        portion_percents = [40, 50, 60]
        for portion_percent in portion_percents:
            size = int(
                percent(portion_percent, candidate_size) + 1
            )
            complement_size = candidate_size + 1 - size
            left_query_portion = rescaled_query_audio[:query_portion_size]
            right_query_portion = rescaled_query_audio[query_portion_size:]
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
        # FIXME: Needs to treat equal distances?
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
    choosed_algorithm, query, candidates_indexes, candidates, original_positions_mapping, use_ls
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
        if (choosed_algorithm in [LINEAR_SCALING, BALS] or use_ls):
            # Rescaling here to optmize time consumption
            rescaled_query_audios = rescale_audio(query_audio)
        else:
            # not an array for jaccard, ra and ktra with use_ls=False
            rescaled_query_audios = query_audio
        for audio_index, candidate_tuple in zip(candidates_indexes, candidates):
            candidate_filename, candidate = candidate_tuple
            candidate = np.array(candidate)
            candidate = np.trim_zeros(candidate)

            if use_ls and choosed_algorithm == KTRA:
                _min_distance, rescaled_query_audios = calculate_linear_scaling(
                    rescaled_query_audios,
                    candidate,
                    include_zero_distance=True
                )
            ##
            distance_or_similarity = matching_algorithms[choosed_algorithm](
                rescaled_query_audios,
                candidate,
                query_audio_name=query_audio_name,
                include_zero_distance=True,  # For LS and BALS
                original_positions_mapping=original_positions_mapping,
                depth=0,  # For recursive alignment and ktra
                k=INITIAL_KTRA_K_VALUE  # For ktra
            )
            ##
            if choosed_algorithm == LINEAR_SCALING:
                query_distance[candidate_filename] = distance_or_similarity[0]
            else:
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

    return all_queries_distances


def calculate_mean_reciprocal_rank(all_queries_distances, results_mapping, show_top_x):
    '''
    Rank results found by apply_matching_algorithm and applies Mean Reciproval
    Rank (MRR).
    '''
    reciprocal_ranks = []
    number_of_queries = len(all_queries_distances.keys())
    for query_name, results in all_queries_distances.items():
        # bounded_results = results[:show_top_x]
        correct_result = results_mapping[query_name]
        # TODO: É realmente sobre todo o dataset? Se não for, ver o que fazer
        # quando o resultado não estiver no bounded_results
        # 'results' is a list of tuples (song_name, distance)
        candidates_names = [result[0] for result in results]
        correct_result_index = candidates_names.index(correct_result)
        rank = correct_result_index + 1
        reciprocal_rank = 1 / rank
        reciprocal_ranks.append(reciprocal_rank)

    mean_reciprocal_rank = sum(reciprocal_ranks) / number_of_queries

    return mean_reciprocal_rank


def calculate_confidence_measurement(
    results, show_top_x, is_training_data=False, results_mapping=None
):
    all_confidence_measurements = {}
    for query_name, result in results.items():
        query_confidence_measurements = []
        bounded_result = result[:show_top_x]
        for index, candidate_and_distance in enumerate(bounded_result):
            candidate, distance = candidate_and_distance
            denominator = [
                _distance
                for candidate_name, _distance in bounded_result
            ]
            denominator.pop(index)
            confidence_measurement = (
                (show_top_x - 1) * distance
            ) / sum(denominator)
            query_confidence_measurements.append(
                (candidate, confidence_measurement)
            )
            # If it's not training data, only calculate confidence for the
            # first result
            if not is_training_data:
                break
        all_confidence_measurements[query_name] = query_confidence_measurements

    if is_training_data:
        train_confidence(all_confidence_measurements, results_mapping)

    return all_confidence_measurements


def _create_index(pitch_contour_segmentations, index_type, num_permutations):
    # Indexing
    td_matrix, audio_mapping, original_positions_mapping = tokenize(
        pitch_contour_segmentations, index_type
    )
    inverted_index = generate_inverted_index(td_matrix, num_permutations)

    # Serializing indexes
    index_type_name = f'inverted_{index_type}'
    indexes_and_indexes_names = [
        (inverted_index, index_type_name),
        # (audio_mapping, 'audio_mapping'),
        # (original_positions_mapping, 'original_positions_mapping')
    ]
    for index, index_name in indexes_and_indexes_names:
        dump_structure(
            index,
            structure_name=index_name
        )


def create_indexes(pitch_contour_segmentations, index_types, num_permutations):
    for index_type in index_types:
        _create_index(pitch_contour_segmentations, index_type, num_permutations)


def _search_in_index(
    query_pitch_contour_segmentations,
    inverted_index,
    songs_list,
    index_type,
    num_permutations
):
    query_td_matrix, _query_audio_mapping, _query_positions_mapping = tokenize(
        query_pitch_contour_segmentations, index_type
    )
    candidates_count = search_inverted_index(
        query_td_matrix, inverted_index, num_permutations
    )
    candidates_indexes = (np.nonzero(candidates_count)[0] - 1)
    candidates = songs_list[candidates_indexes]

    return candidates_indexes, candidates


def search(
    query_pitch_contour_segmentations,
    song_pitch_contour_segmentations,
    index_types,
    matching_algorithm,
    use_ls,
    show_top_x,
    is_training_confidence,
    num_permutations,
    results_mapping=None
):
    # Recovering songs pitch vectors
    song_pitch_vectors = unzip_pitch_contours(
        song_pitch_contour_segmentations
    )

    # Recovering query pitch vectors
    query_pitch_vectors = unzip_pitch_contours(
        query_pitch_contour_segmentations
    )
    results = None
    for index_type in index_types:
        # Recovering dumped index
        try:
            original_positions_mapping = None
            # inverted_index, audio_mapping, original_positions_mapping = (
            #     load_structure(structure_name=index_name)
            #     for index_name in [
            #         f'inverted_{index_type}',
            #         # 'audio_mapping',
            #         # 'original_positions_mapping'
            #     ]
            # )
            inverted_index_name = f'inverted_{index_type}'
            inverted_index = load_structure(inverted_index_name)
        except Exception as e:
            log_no_dumped_files_error(e)
            exit(1)

        # Searching songs
        candidates_indexes, candidates = _search_in_index(
            query_pitch_contour_segmentations,
            inverted_index=inverted_index,
            songs_list=song_pitch_vectors,
            index_type=index_type,
            num_permutations=num_permutations
        )

        results = apply_matching_algorithm(
            choosed_algorithm=matching_algorithm,
            query=query_pitch_vectors,
            candidates_indexes=candidates_indexes,
            candidates=candidates,
            original_positions_mapping=original_positions_mapping,
            use_ls=use_ls
        )

        all_confidence_measurements = calculate_confidence_measurement(
            results=results,
            show_top_x=show_top_x,
            is_training_data=is_training_confidence,
            results_mapping=results_mapping
        )

        if not is_training_confidence:
            print('TO BE IMPLEMENTED YET')
            # print_confidence_measurements(all_confidence_measurements)
            # TODO: get confidence from file
            # TODO: Stop if confidence measurement is higher than the threshold
            # TODO: The threshold should be automatically optimized so
            # that the true candidates are returned directly and the clip
            # with the false first ranked candidate is put into the next
            # filter. In the QBH system, we randomly select part of
            # query clips as the training set to obtain the threshold

    return results
