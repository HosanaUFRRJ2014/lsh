# -*-coding:utf8;-*-
import sys
import numpy as np
from argparse import ArgumentParser

from constants import (
    DEFAULT_NUMBER_OF_PERMUTATIONS,
    SERIALIZE_PITCH_VECTORS,
    CREATE_INDEX,
    INDEX_TYPES,
    SEARCH,
    SEARCH_ALL,
    SEARCH_METHODS,
    REQUIRE_INDEX_TYPE,
    METHODS,
    LINEAR_SCALING,
    RECURSIVE_ALIGNMENT,
    KTRA,
    MATCHING_ALGORITHMS,
    SHOW_TOP_X
)
from json_manipulator import (
    load_structure,
    serialize_pitch_contour_segmentations,
    deserialize_songs_pitch_contour_segmentations,
    deserialize_queries_pitch_contour_segmentations
)
from lsh import (
    apply_matching_algorithm_to_lsh,
    create_indexes,
    search_indexes,
    calculate_mean_reciprocal_rank
)
from loader import (
    load_audio_pitch_contour_segmentation,
    load_expected_results
)
from utils import (
    is_create_index_or_search_method,
    print_results,
    validate_program_args
)


def process_args():
    '''
    Processes program args.
    Returns a tuple containing the processed program args.
    '''
    no_need_to_inform_warning = "Don't inform if you want the default value."
    parser = ArgumentParser()
    help_msg = "".join([
        "Number of permutations LSH will perform.",
        f" Defaults to {DEFAULT_NUMBER_OF_PERMUTATIONS}."
    ])
    parser.add_argument(
        "method",
        type=str,
        help="Options: {}".format(
            ', '.join(METHODS)
        )
    )
    parser.add_argument(
        "--index_types",
        "-i",
        nargs='+',
        type=str,
        help="Options: {}. Inform at least one if \"method\" is any of \"{}\".".format(
            ', '.join(INDEX_TYPES),
            ', '.join(REQUIRE_INDEX_TYPE)
        ),
        required=is_create_index_or_search_method(sys.argv)
    )
    parser.add_argument(
        "--query_filename",
        "-f",
        type=str,
        help="Filename of a .wav file to search. NOTE: Loads pitches before search.",
        default=''
    )
    parser.add_argument(
        "--num_permutations",
        "-np",
        type=int,
        help=help_msg,
        default=DEFAULT_NUMBER_OF_PERMUTATIONS
    )
    parser.add_argument(
        "--show_top",
        "-top",
        type=int,
        help=f"Shows top X results. Defaults to {SHOW_TOP_X}.",
        default=SHOW_TOP_X
    )
    parser.add_argument(
        "--matching_algorithm",
        "-ma",
        type=str,
        help=f"It's expected to be informed alongside {SEARCH_METHODS} methods. "
        + "Options: {}. Defaults to {}".format(
            ', '.join(MATCHING_ALGORITHMS),
            LINEAR_SCALING
        ),
        default=LINEAR_SCALING
    )
    parser.add_argument(
        "--use_ls",
        type=bool,
        help=' '.join([
            f"If {RECURSIVE_ALIGNMENT} and {KTRA} will include {LINEAR_SCALING}.",
            f"Defaults to False. {no_need_to_inform_warning}"
        ]),
        default=False
    )
    parser.add_argument(
        "--train_confidence",
        type=bool,
        help=" ".join([
            f"If \"True\" and informed alongside {SEARCH_ALL} method,",
            "confidence measurement will be in training mode,",
            "which means that confidence of all the top X candidates",
            "will be calculated.",
            "If \"False\", only the measurement of the first result will be",
            f"calculated. Defaults to False. {no_need_to_inform_warning}"
        ]),
        default=False
    )
    parser.add_argument(
        "--num_audios",
        type=int,
        help=' '.join([
            f"If especified, limits number of audios for {CREATE_INDEX} and",
            f"{SEARCH_ALL} methods.",
            "Gets the first 'num_audios' audios.",
            "Defaults to None."
        ]),
        default=None
    )
    args = parser.parse_args()
    num_permutations = args.num_permutations
    method_name = args.method
    index_types = args.index_types if args.index_types else []
    query_filename = args.query_filename
    matching_algorithm = args.matching_algorithm
    use_ls = args.use_ls
    show_top_x = args.show_top
    is_training_confidence = args.train_confidence
    num_audios = args.num_audios

    # Validate args. If any of them is invalid, exit program.
    validate_program_args(
        method_name=method_name,
        matching_algorithm=matching_algorithm,
        index_types=index_types,
        is_training_confidence=is_training_confidence
    )

    return method_name, \
        index_types, \
        query_filename, \
        num_permutations, \
        matching_algorithm, \
        use_ls, \
        show_top_x, \
        is_training_confidence, \
        num_audios


def main():
    method_name,  \
        index_types,  \
        query_filename,  \
        num_permutations,  \
        matching_algorithm,  \
        use_ls,  \
        show_top_x, \
        is_training_confidence, \
        num_audios = process_args()

    if method_name == SERIALIZE_PITCH_VECTORS:
        serialize_pitch_contour_segmentations()
    elif method_name == CREATE_INDEX:
        # Creating index(es)
        pitch_contour_segmentations = deserialize_songs_pitch_contour_segmentations(num_audios)
        create_indexes(
            pitch_contour_segmentations=pitch_contour_segmentations,
            index_types=index_types,
            num_permutations=num_permutations
        )
    elif method_name in SEARCH_METHODS:
        # Loading query and song pitch vectors
        if method_name == SEARCH:
            query_pitch_contour_segmentations = load_audio_pitch_contour_segmentation(query_filename)
        elif method_name == SEARCH_ALL:
            query_pitch_contour_segmentations = deserialize_queries_pitch_contour_segmentations(num_audios)

        song_pitch_contour_segmentations = deserialize_songs_pitch_contour_segmentations()

        # Searching and applying matching algorithms
        results_mapping = load_expected_results()
        results = search_indexes(
            query_pitch_contour_segmentations=query_pitch_contour_segmentations,
            song_pitch_contour_segmentations=song_pitch_contour_segmentations,
            index_types=index_types,
            matching_algorithm=matching_algorithm,
            use_ls=use_ls,
            show_top_x=show_top_x,
            is_training_confidence=is_training_confidence,
            num_permutations=num_permutations,
            results_mapping=results_mapping
        )

        # Results and MRR
        print_results(matching_algorithm, index_types, results, show_top_x)
        mrr = calculate_mean_reciprocal_rank(results, results_mapping, show_top_x)

        print('Mean Reciprocal Ranking (MRR): ', mrr)


if __name__ == '__main__':
    main()


# TODO: Renomear pitch_vectors, posto que estes contém o audio path e
# o vetor de pitches
