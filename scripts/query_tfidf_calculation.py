

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

from constants import SONG
from json_manipulator import (
    deserialize_songs_pitch_contour_segmentations,
    dump_structure,
    load_structure,
    get_songs_count
)

from loader import load_expected_results


def get_vocabulary(data_frame):
    return data_frame.columns.to_numpy()


def obtain_query_remaining_pitches_and_tfidfs(**kwargs):
    query_filename = kwargs.get("filename")
    results_mapping = kwargs.get("results_mapping")
    song_filename = results_mapping.get(query_filename)
    song_tfidfs = kwargs.get("tfidfs").get(song_filename)
    original_vector = kwargs.get("original_vector")
    vocabulary_remaining_pitches = kwargs.get("vocabulary_remaining_pitches")

    # Gets corresponding query TF-IDFs and remaining pitches
    remaining_pitches = []
    query_tfidfs = {}

    for pitch in original_vector:
        if pitch in vocabulary_remaining_pitches:
            remaining_pitches.append(pitch)
            pitch_tfidf = song_tfidfs.get(str(pitch))
            if pitch_tfidf:
                query_tfidfs[pitch] = pitch_tfidf

    return remaining_pitches, query_tfidfs


def extract_remaining_pitches(tfidfs, all_pitch_contour_segmentations, num_audios, min_tfidf, audio_type, vocabulary_remaining_pitches):
    percentages = {}
    queries_tfidfs = {}
    no_remaining_pitches_count = 0
    all_remaining_pitches = {}

    results_mapping = load_expected_results()
    kwargs = {
        "tfidfs": tfidfs,
        "min_tfidf": min_tfidf,
        "vocabulary_remaining_pitches": vocabulary_remaining_pitches,
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

        remaining_pitches, query_tfidfs = obtain_query_remaining_pitches_and_tfidfs(
            **kwargs
        )
        queries_tfidfs[filename] = query_tfidfs

        # ignore audios without remainings
        if len(remaining_pitches) == 0:
            no_remaining_pitches_count += 1
            continue

        # calculate percentage of remainings
        original_length = len(original_pitch_vector)
        remaining_length = len(remaining_pitches)

        percent = (remaining_length/original_length) * 100
        percentages[filename] = percent

        all_remaining_pitches[filename] = remaining_pitches

        print("Original size:", original_length)
        print("Remaining size:", remaining_length)
        print("Percent:", percent, " %", "\n")

    no_pitches_percent = (no_remaining_pitches_count/num_audios) * 100

    return all_remaining_pitches, percentages, no_pitches_percent, queries_tfidfs


def main():
    tfidf_data_frame = load_structure(
        structure_name=f'{SONG}_tf_idfs_per_file',
        as_numpy=False,
        as_pandas=True,
        extension='pkl'
    )
    vocabulary = get_vocabulary(tfidf_data_frame)

    all_remaining_pitches, percentages, no_pitches_percent, queries_tfidfs = extract_remaining_pitches(
        tfidf_data_framed,
        all_pitch_contour_segmentations,
        num_audios,
        min_tfidf=min_tfidf,
        audio_type=audio_type,
        vocabulary_remaining_pitches=vocabulary_remaining_pitches  # empty if audio type is song
    )


if __name__ == "__main__":
    main()
