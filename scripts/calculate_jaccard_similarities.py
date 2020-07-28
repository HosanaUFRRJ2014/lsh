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

from json_manipulator import (
    deserialize_songs_pitch_contour_segmentations,
    deserialize_queries_pitch_contour_segmentations,
    dump_structure,
    load_structure
)

from matching_algorithms import calculate_jaccard_similarity
from loader import load_expected_results
from loader import get_songs_count
from constants import SONG, QUERY


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
        "--min_tfidf",
        "-min",
        type=float,
        help=f"Calculates jaccard similarity for this minimum TF-IDF. If not informed, considers all pitches."
    )

    args = parser.parse_args()

    # num_audios = args.num_audios
    min_tfidf = args.min_tfidf
    # return num_audios, min_tfidf
    return min_tfidf


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


def main():
    jaccard_similarities = {}
    # num_audios, min_tfidf = process_args
    min_tfidf = process_args()
    results_mapping = load_expected_results()

    if min_tfidf:
        songs, queries = load_remainings(min_tfidf)
        structure_name = f"jaccard_similarities_min_tfidf_{min_tfidf}"
    else:
        songs, queries = load_originals()
        structure_name = f"jaccard_similarities"

    for query_filename, query_pitches_values in queries.items():
        expected_song_name = results_mapping[query_filename]
        expected_song = songs[expected_song_name]

        # Apply jaccard
        jaccard_similarities[query_filename] = calculate_jaccard_similarity(
            query_audio=query_pitches_values,
            candidate=expected_song
        )

    dump_structure(
        structure_name=structure_name,
        structure=jaccard_similarities
    )

if __name__ == "__main__":
    main()
