from sys import float_info
from essentia.standard import Mean
import numpy as np
from scipy.ndimage.interpolation import shift
from constants import (
    PLSH_INDEX,
    NLSH_INDEX,
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
from utils import percent


__all__ = ["apply_matching_algorithm"]

MAX_FLOAT = float_info.max
mean = Mean()


def _mean_substract(pitch_vector):
    return pitch_vector - mean.compute(pitch_vector)


def _calculate_jaccard_similarity(query_audio, candidate, **kwargs):
    '''
    Jaccard Similarity algorithm in steps:
    1. Get the shared members between both sets, i.e. intersection.
    2. Get the members in both sets (shared and un-shared, i.e. union).
    3. Divide the number of shared members found in (1) by the total number of
       members found in (2).
    4. Multiply the found number in (3) by 100.
    '''
    from lsh import (
        exec_plsh_pitch_extraction as extract_plsh_pitches,
        exec_nlsh_pitch_extraction as extract_nlsh_pitches
    )  # circular import
    index_type = kwargs.get('index_type')

    exec_pitch_extraction = {
        PLSH_INDEX: extract_plsh_pitches,
        NLSH_INDEX: extract_nlsh_pitches
    }
    # import ipdb; ipdb.set_trace()
    query_chunks = exec_pitch_extraction[index_type](query_audio.tolist())
    candidate_chunks = exec_pitch_extraction[index_type](candidate.tolist())

    intersection = np.intersect1d(candidate_chunks, query_chunks)
    union = np.union1d(candidate_chunks, query_chunks)

    jaccard_similarity = 0
    if union.size > 0:
        jaccard_similarity = (intersection.size / union.size) * 100

    return jaccard_similarity


def _rescale_audio(query_audio):
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


def _calculate_manhattan_distance(rescaled_audio, candidate):
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


def _calculate_linear_scaling(
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
        distance = _calculate_manhattan_distance(
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


def _get_candidate_neighbourhood(**kwargs):
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


def _calculate_bals(rescaled_query_audios, candidate, **kwargs):
    '''
    Explore candidates neighbourhood
       - For each candidate, lenghten or shorten it.
    For each neighbor, measure LS distance.
    Retain the fragment with the shortest distance.
    '''
    candidate_distance, _query = _calculate_linear_scaling(
        rescaled_query_audios, candidate, include_zero_distance=True
    )

    kwargs['candidate'] = candidate

    neighbours = _get_candidate_neighbourhood(**kwargs)
    # Starts with the similar audio
    nearest_neighbour_distance = candidate_distance
    for neighbour in neighbours:
        distance, _query = _calculate_linear_scaling(
            rescaled_query_audios,
            neighbour,
            include_zero_distance=True
        )
        if distance < nearest_neighbour_distance:
            nearest_neighbour_distance = distance

    return nearest_neighbour_distance


def _recursive_align(query_audio, candidate, **kwargs):
    # Compute the linear distance of the corresponding part
    min_distance, rescaled_query_audio = _calculate_linear_scaling(
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

            left_distance = _recursive_align(
                left_query_portion,
                left_similar_portion,
                depth=depth + 1
            )

            right_distance = _recursive_align(
                right_query_portion,
                right_similar_portion,
                depth=depth + 1
            )

            min_distance = min([left_distance, right_distance, min_distance])

    return min_distance


def _calculate_ktra(query_audio, candidate, **kwargs):
    depth = kwargs.get('depth')

    if depth == 0:
        query_audio = _mean_substract(query_audio)
        candidate = _mean_substract(candidate)

    k = kwargs.get('k')

    d_minus = _recursive_align(query_audio - k, candidate, depth=0)
    d_zero = _recursive_align(query_audio, candidate, depth=0)
    d_plus = _recursive_align(query_audio + k, candidate, depth=0)

    min_distance = min([d_minus, d_zero, d_plus])
    if depth < MAX_KTRA_DEPTH:
        # FIXME: Needs to treat equal distances?
        if d_minus == min_distance:
            query_audio = query_audio - k
        elif d_plus == min_distance:
            query_audio = query_audio + k

        min_distance = _calculate_ktra(
            query_audio,
            candidate,
            k=k / 2,
            depth=depth + 1
        )

    return min_distance


def apply_matching_algorithm(
    choosed_algorithm, query, candidates, index_type, original_positions_mapping, use_ls
):
    matching_algorithms = {
        JACCARD_SIMILARITY: _calculate_jaccard_similarity,
        LINEAR_SCALING: _calculate_linear_scaling,
        BALS: _calculate_bals,
        RECURSIVE_ALIGNMENT: _recursive_align,
        KTRA: _calculate_ktra
    }

    all_queries_distances = {}
    for query_audio_name, query_audio in query:
        query_audio = np.array(query_audio)
        query_audio = np.trim_zeros(query_audio)
        query_distance = dict()
        if (choosed_algorithm in [LINEAR_SCALING, BALS] or use_ls):
            # Rescaling here to optmize time consumption
            rescaled_query_audios = _rescale_audio(query_audio)
        else:
            # not an array for jaccard, ra and ktra with use_ls=False
            rescaled_query_audios = query_audio
        for candidate_tuple in candidates:
            candidate_filename, candidate = candidate_tuple
            candidate = np.array(candidate)
            candidate = np.trim_zeros(candidate)

            if use_ls and choosed_algorithm == KTRA:
                _min_distance, rescaled_query_audios = _calculate_linear_scaling(
                    rescaled_query_audios,
                    candidate,
                    include_zero_distance=True
                )
            ##
            distance_or_similarity = matching_algorithms[choosed_algorithm](
                rescaled_query_audios,
                candidate,
                query_audio_name=query_audio_name,
                index_type=index_type,  # For Jaccard
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
