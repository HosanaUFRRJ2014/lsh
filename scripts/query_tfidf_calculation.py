import os.path
import sys
import textwrap
import numpy as np
import pandas as pd
from argparse import (
    ArgumentParser,
    RawDescriptionHelpFormatter
)
from math import floor, log2
# Includes the parent directory into sys.path, to make imports work
sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)
        ),
        os.pardir
    )
)

from constants import (
    SONG,
    AUDIO_TYPES,
    QUERY
)
from json_manipulator import (
    deserialize_queries_pitch_contour_segmentations,
    dump_structure,
    load_structure,
    get_queries_count
)

from loader import load_expected_results


def process_args():
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
    )

    default_save_graphic = True

    default_num_songs = get_queries_count()

    parser.add_argument(
        "--num_songs",
        "-na",
        type=int,
        help=" ".join([
            "Number of songs to consider. If not informed,",
            "will apply calculations only considering MIREX datasets."
        ]),
        default=default_num_songs
    )

    parser.add_argument(
        "--save_graphic",
        type=bool,
        help=" ".join([
            "If True, saves remainings percentages graphic",
            f"Defaults to {default_save_graphic}."
        ]),
        default=default_save_graphic
    )

    args = parser.parse_args()

    num_songs = args.num_songs
    will_save_graphic = args.save_graphic

    return num_songs, will_save_graphic


def get_vocabulary(data_frame):
    return data_frame.columns.values


def get_filenames(data_frame):
    return data_frame.index.values


def estimate_query_tfidfs(**kwargs):
    query_filename = kwargs.get("filename")
    results_mapping = kwargs.get("results_mapping")
    song_filename = results_mapping.get(query_filename)
    tfidfs = kwargs.get("tfidfs")
    original_vector = kwargs.get("original_vector")
    vocabulary = kwargs.get("vocabulary")


    # Gets corresponding query TF-IDFs
    query_tfidfs = {}

    for pitch in original_vector:
        if pitch in vocabulary:
            # Gets the max value of tfdif from vocabulary, so the pitch will
            # have the slightest chance to be removed in query 
            pitch_tfidf = np.max(tfidfs.get(pitch))
            query_tfidfs[pitch] = pitch_tfidf

            # pitch_tfidf = tfidfs.at[song_filename, pitch]
            # if pitch_tfidf: # and not np.isnan(pitch_tfidf):

    return query_tfidfs


def estimate_queries_tfidfs(tfidfs, all_pitch_contour_segmentations, vocabulary):
    queries_tfidfs = {}

    results_mapping = load_expected_results()
    kwargs = {
        "tfidfs": tfidfs,
        "vocabulary": vocabulary,
        "results_mapping": results_mapping
    }
    for filename, original_pitch_vector, _onsets, _durations in all_pitch_contour_segmentations:
        # Removes zeros values
        # original_pitch_vector = np.array(original_pitch_vector)
        # original_pitch_vector = original_pitch_vector[np.nonzero(original_pitch_vector)[0]]

        # If query, gets its corresponding song. Otherwise, it's the audio name. 
        # filename = results_mapping.get(filename, filename)
        
        kwargs["filename"] = filename
        kwargs["original_vector"] = original_pitch_vector

        query_tfidfs = estimate_query_tfidfs(
            **kwargs
        )
        queries_tfidfs[filename] = query_tfidfs

    return queries_tfidfs


def main():
    num_songs, will_save_graphic = process_args()

    tfidf_data_frame = load_structure(
        structure_name=f"{num_songs}_songs/{SONG}_tf_idfs_per_file",
        as_numpy=False,
        as_pandas=True,
        extension='pkl'
    )
    vocabulary = get_vocabulary(tfidf_data_frame)

    all_pitch_contour_segmentations = deserialize_queries_pitch_contour_segmentations()

    queries_tfidfs = estimate_queries_tfidfs(
        tfidf_data_frame,
        all_pitch_contour_segmentations,
        vocabulary=vocabulary
    )

    data_frame = pd.DataFrame.from_dict(
        queries_tfidfs,
        orient='index',
        dtype=np.float64
    )

    dump_structure(
        data_frame,
        structure_name=f"{num_songs}_songs/{QUERY}_tf_idfs_per_file",
        as_numpy=False,
        as_pandas=True,
        extension="pkl"
    )    


if __name__ == "__main__":
    main()
