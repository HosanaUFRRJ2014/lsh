# -*-coding:utf8;-*-
from copy import copy
from math import floor, ceil
import numpy as np
from scipy.sparse import dok_matrix

from constants import (
    PLSH_INDEX,
    NLSH_INDEX,
    SELECTION_FUNCTIONS,
    SELECTION_FUNCTION_COUNT
)
from json_manipulator import (
    dump_structure,
    load_structure,
    NumpyArrayEncoder
)
from loader import load_expected_results
from matching_algorithms import apply_matching_algorithm
from messages import (
    log_could_not_calculate_mrr_warning,
    log_no_dumped_files_error
)

from utils import (
    get_confidence_measurement,
    load_sparse_matrix,
    print_confidence_measurements,
    save_sparse_matrix,
    train_confidence,
    unzip_pitch_contours
)


__all__ = [
    "create_indexes",
    "exec_nlsh_pitch_extraction",
    "exec_plsh_pitch_extraction",
    "search_indexes"
]


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
    WINDOW_SHIFT = 1 # 3   # in article: 1
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
            # audio_map = _audio_mapping_index(dumped_piece, audio_map, filename)
            # orig_pos_map = _original_position_index(
            #     dumped_piece,
            #     orig_pos_map,
            #     audio_chunk_index,
            #     filename
            # )
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

    matrix_of_indexes = dok_matrix(
        (num_lines, num_columns), dtype=np.int
    )

    # First position will never be occupied, in order to difference 0 from
    # a real position in matrix of indexes
    EMPTY_ARRAY = np.array([])
    inverted_index_data = np.array([EMPTY_ARRAY], dtype=np.ndarray)

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
                if non_zero_indexes[0].size > 0:
                    second_index = int(
                        SELECTION_FUNCTIONS[l](dj_permutation[non_zero_indexes])
                    ) - 1

                    data_position = matrix_of_indexes[first_index, second_index]
                    is_empty_position = data_position == 0

                    if is_empty_position:
                        # Appends np.array([j+1]) in inverted_index_data
                        inverted_index_data = np.append(
                            inverted_index_data,
                            [j + 1]
                        )
                        # Gets its position and updates matrix with it
                        new_data_index = inverted_index_data.size - 1
                        matrix_of_indexes[first_index, second_index] = new_data_index
                    else:
                        # Retrieve array from inverted_index_data
                        array_to_update = inverted_index_data[data_position]
                        # Appends [j+1] to it
                        updated_array = np.append(
                            array_to_update,
                            [j + 1]
                        )
                        # update inverted_index_data with the updated_array
                        inverted_index_data[data_position] = updated_array
    
    return matrix_of_indexes, inverted_index_data


def search_inverted_index(
    query_td_matrix, matrix_of_indexes, inverted_index_data, permutation_count
):
    # num_lines = permutation_count * (SELECTION_FUNCTION_COUNT + 1) + SELECTION_FUNCTION_COUNT
    candidates_count = np.zeros((matrix_of_indexes.shape[1] + 1, ), dtype=int)
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
                        data_position = matrix_of_indexes[first_index, second_index]
                        is_filled_position = data_position != 0
                        if is_filled_position:
                            retrieved_pitch_vector = inverted_index_data[data_position]
                            candidates_count[retrieved_pitch_vector] += 1
                    except IndexError as e:
                        continue
    return candidates_count


def calculate_mean_reciprocal_rank(all_queries_distances, results_mapping, show_top_x):
    '''
    Rank results found by apply_matching_algorithm and applies Mean Reciproval
    Rank (MRR).
    '''
    reciprocal_ranks = []
    mean_reciprocal_rank = None
    number_of_queries = len(all_queries_distances.keys())
    for query_name, results in all_queries_distances.items():
        # bounded_results = results[:show_top_x]
        correct_result = results_mapping[query_name]
        # TODO: É realmente sobre todo o dataset? Se não for, ver o que fazer
        # quando o resultado não estiver no bounded_results
        # 'results' is a list of tuples (song_name, distance)
        candidates_names = [result[0] for result in results]

        try:
            correct_result_index = candidates_names.index(correct_result)
            rank = correct_result_index + 1
            reciprocal_rank = 1 / rank
        except ValueError:
            log_could_not_calculate_mrr_warning(query_name)
            reciprocal_rank = None
            exit(1)

        reciprocal_ranks.append(reciprocal_rank)

    if all(reciprocal_ranks):
        mean_reciprocal_rank = sum(reciprocal_ranks) / number_of_queries

    return mean_reciprocal_rank


def calculate_confidence_measurement(
    results, show_top_x, is_training_data=False, results_mapping=None
):
    all_confidence_measurements_data = {}
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
        all_confidence_measurements_data[query_name] = query_confidence_measurements

    if is_training_data:
        train_confidence(all_confidence_measurements_data, results_mapping)

    return all_confidence_measurements_data


def clip_false_candidates(all_confidence_measurements_data):
    """
    The clip with the false first ranked candidate is put into the next filter.
    """
    removed_candidates = []
    above_threshold_count = 0
    candidate_confidence_measurement = [
        data[0]
        for data in list(
            all_confidence_measurements_data.values()
        )
    ][0]

    threshold = get_confidence_measurement()

    measurement = candidate_confidence_measurement[-1]
    if measurement > threshold:
        above_threshold_count += 1
    else:
        removed_candidates.append(
            candidate_confidence_measurement[0]
        )

    no_need_of_second_filter = above_threshold_count == len(
        all_confidence_measurements_data
    )

    return removed_candidates, no_need_of_second_filter


def _create_index(pitch_contour_segmentations, index_type, num_permutations):
    # Indexing
    print('Tokenizing... (step 1/3)')
    td_matrix, audio_mapping, original_positions_mapping = tokenize(
        pitch_contour_segmentations, index_type
    )
    print('Generating inverted index... (step 2/3)')
    matrix_of_indexes, inverted_index_data = generate_inverted_index(td_matrix, num_permutations)

    # Serializing indexes
    index_type_name = f'inverted_{index_type}_data'
    matrix_of_indexes_name = f'matrix_of_{index_type}'
    indexes_and_indexes_names = [
        (inverted_index_data, index_type_name),
        # (audio_mapping, 'audio_mapping'),
        # (original_positions_mapping, 'original_positions_mapping')
    ]
    print(
        f'Saving {index_type_name} and {matrix_of_indexes_name}',
        ' in files (step 3/3)'
    )
    save_sparse_matrix(matrix_of_indexes, structure_name=matrix_of_indexes_name)
    for index, index_name in indexes_and_indexes_names:
        dump_structure(
            index,
            structure_name=index_name
        )


def create_indexes(pitch_contour_segmentations, index_types, num_permutations):
    for index_type in index_types:
        _create_index(pitch_contour_segmentations, index_type, num_permutations)


def _search_index(
    query_pitch_contour_segmentations,
    matrix_of_indexes,
    inverted_index_data,
    songs_list,
    index_type,
    removed_candidates,
    num_permutations
):
    print('Tokenizing... (step 2/5)')
    query_td_matrix, _query_audio_mapping, _query_positions_mapping = tokenize(
        query_pitch_contour_segmentations, index_type
    )
    print('Searching in inverted index... (step 3/5)')
    candidates_count = search_inverted_index(
        query_td_matrix, matrix_of_indexes, inverted_index_data, num_permutations
    )
    candidates_indexes = (np.nonzero(candidates_count)[0] - 1)
    candidates = songs_list[candidates_indexes]

    # Clip with the false first ranked candidates is put into the next filter.
    if removed_candidates:
        candidates = np.array([
            candidate
            for candidate in candidates
            if candidate[0] not in removed_candidates
        ])
    return candidates


def search_indexes(
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
    print('Retrieving data... (step 1/5)')
    # Recovering songs pitch vectors
    song_pitch_vectors = unzip_pitch_contours(
        song_pitch_contour_segmentations
    )

    # Recovering query pitch vectors
    query_pitch_vectors = unzip_pitch_contours(
        query_pitch_contour_segmentations
    )
    results = None
    removed_candidates = []
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
            matrix_of_index_name = f'matrix_of_{index_type}'
            inverted_index_name = f'inverted_{index_type}_data'

            matrix_of_indexes = load_sparse_matrix(matrix_of_index_name)
            inverted_index_data = load_structure(inverted_index_name)
        except Exception as e:
            log_no_dumped_files_error(e)
            exit(1)

        # Searching songs
        candidates = _search_index(
            query_pitch_contour_segmentations,
            matrix_of_indexes=matrix_of_indexes,
            inverted_index_data=inverted_index_data,
            songs_list=song_pitch_vectors,
            index_type=index_type,
            removed_candidates=removed_candidates,
            num_permutations=num_permutations
        )

        print('Applying matching algorithm... (step 4/5)')
        results = apply_matching_algorithm(
            choosed_algorithm=matching_algorithm,
            query=query_pitch_vectors,
            candidates=candidates,
            index_type=index_type,
            original_positions_mapping=original_positions_mapping,
            use_ls=use_ls
        )

        all_confidence_measurements_data = calculate_confidence_measurement(
            results=results,
            show_top_x=show_top_x,
            is_training_data=is_training_confidence,
            results_mapping=results_mapping
        )

        if not is_training_confidence:
            removed_candidates, all_passed = clip_false_candidates(
                all_confidence_measurements_data
            )
            if all_passed or len(index_types) == 1:
                print("There is no need of a second filter")
                break

    return results
