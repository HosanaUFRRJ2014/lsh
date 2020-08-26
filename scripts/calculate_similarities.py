# Includes the parent directory into sys.path, to make imports work
import os.path, sys
sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)
        ),
        os.pardir
    )
)

from argparse import ArgumentParser
import numpy as np
import pandas as pd

from json_manipulator import (
    deserialize_songs_pitch_contour_segmentations,
    deserialize_queries_pitch_contour_segmentations,
    dump_structure,
    load_structure
)

from matching_algorithms import apply_matching_algorithm_to_tfidf
from loader import load_expected_results
from loader import get_songs_count
from constants import (
    SONG,
    QUERY,
    SEARCH_METHODS,
    MATCHING_ALGORITHMS,
    JACCARD_SIMILARITY,
    COSINE_SIMILARITY
)
from utils import is_invalid_matching_algorithm
from messages import (
    log_invalid_matching_algorithm_error,
    log_useless_arg_warn
)

def process_args():
    parser = ArgumentParser()

    default_num_audios = get_songs_count()
    default_plot_tfdfs = False
    default_plot_remaining_percents = True

    # parser.add_argument(
    #     "--num_audios",
    #     "-na",
    #     type=int,
    #     help=" ".join([
    #         "Number of audios to consider. If not informed,",
    #         "will apply calculations for the entire dataset."
    #     ]),
    #     default=default_num_audios
    # )

    parser.add_argument(
        "--matching_algorithm",
        "-ma",
        type=str,
        help="Options: {}. Defaults to {}".format(
            ', '.join(MATCHING_ALGORITHMS),
            JACCARD_SIMILARITY
        ),
        default=JACCARD_SIMILARITY
    )

    parser.add_argument(
        "--min_tfidf",
        "-min",
        type=float,
        help=f"Calculates similarity for this minimum TF-IDF. If not informed, considers all pitches."
    )

    args = parser.parse_args()

    # num_audios = args.num_audios
    min_tfidf = args.min_tfidf if args.min_tfidf else 0.0
    matching_algorithm = args.matching_algorithm
    is_invalid_matching_algorithm(matching_algorithm)

    # return num_audios, min_tfidf
    return min_tfidf, matching_algorithm


def load_originals():     
    all_songs_pitch_contour_segmentations = deserialize_songs_pitch_contour_segmentations()
    all_queries_pitch_countour_segmentations = deserialize_queries_pitch_contour_segmentations()

    songs_filenames, songs_pitches_values, _, _ = zip(
        *all_songs_pitch_contour_segmentations
    )
    songs_original_pitches = dict(zip(songs_filenames, songs_pitches_values))

    queries_filenames, queries_pitches_values, _, _ = zip(
        *all_queries_pitch_countour_segmentations
    )
    queries_original_pitches = dict(zip(queries_filenames, queries_pitches_values))

    return songs_original_pitches, queries_original_pitches


def load_remainings(min_tfidf):
    # TODO: Add custom try/except error message here
    songs_remainings_pitches = load_structure(
        structure_name=f'remaining_pitches_min_tfidf_{min_tfidf}_per_{SONG}',
        as_numpy=False
    )
    queries_remaining_pitches = load_structure(
        structure_name=f'remaining_pitches_min_tfidf_{min_tfidf}_per_{QUERY}',
        as_numpy=False
    )
    return songs_remainings_pitches, queries_remaining_pitches


def main():
    similarities = {}

    min_tfidf, matching_algorithm = process_args()
    results_mapping = load_expected_results()
    songs_tfidfs = load_structure(
        structure_name=f'{SONG}_tf_idfs_per_file',
        as_numpy=False,
        as_pandas=True,
        extension="pkl"
    )

    queries_tfidfs = load_structure(
        structure_name=f'{QUERY}_tf_idfs_per_file',
        as_numpy=False,
        as_pandas=True,
        extension="pkl"
    )

    if min_tfidf:
        songs, queries = load_remainings(min_tfidf)
        structure_name = f"{matching_algorithm}_min_tfidf_{min_tfidf}"
    else:
        songs, queries = load_originals()
        structure_name = f"{matching_algorithm}"


    for query_filename, query_pitches_values in queries.items():
        expected_song_filename = results_mapping[query_filename]
        expected_song = songs[expected_song_filename]

        if matching_algorithm == COSINE_SIMILARITY:
            # ensures song and query vectors will have the same size
            merged_dataframe = pd.DataFrame.from_records(
                [
                    songs_tfidfs.loc[expected_song_filename],
                    queries_tfidfs.loc[query_filename]
                ],
                index=[expected_song_filename, query_filename]
            ) 

            # replaces nan by zero
            merged_dataframe = merged_dataframe.fillna(0)

            # Filter out tfidfs less than min-tfidf
            merged_dataframe = merged_dataframe.where(
                merged_dataframe > min_tfidf,
                other=0
            )

            kwargs = {
                "song_tfidfs": merged_dataframe.loc[expected_song_filename], 
                "query_tfidfs": merged_dataframe.loc[query_filename]
            }
        else:
            kwargs = {
                "query": np.array(query_pitches_values),
                "song": np.array(expected_song),
            }

        similarities[query_filename] = apply_matching_algorithm_to_tfidf(
            choosed_algorithm=matching_algorithm,
            **kwargs
        )

    path = "similarities"
    dump_structure(
        structure_name=f"{path}/{structure_name}",
        structure=similarities
    )

if __name__ == "__main__":
    main()


# - [REVIEW] Criar passo para "cálculo" de tfidf de queries, para substituir ou antes de tfidf extraction 
# - [REVIEW] Modificar obtain remaining pitches de query_tfidf_calculation, para poder pegar
# os dados do dataframe
# -  TODO: Filtrar música e candidatos para excluir os min-tfidf menor que um dado threshold
# em calculate_similarities
# 
