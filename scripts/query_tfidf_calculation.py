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
    # description = textwrap.dedent('''\
    # Extracts pitches above the min_tfidf threshold.
    # Saves:
    #     - Remaining pitches separated by file (songs and queries filenames),
    #     - Set of remaining pitches,
    #     - etc
    # ''')
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        # description=description
    )

    default_min_tfidf = 0.01
    default_save_graphic = True

    parser.add_argument(
        "--num_audios",
        "-na",
        type=int,
        help=" ".join([
            "Number of audios to consider. If not informed,",
            "will apply calculations for the entire dataset."
        ]),
        # default=default_num_audios
    )

    parser.add_argument(
        "--min_tfidf",
        "-min",
        type=float,
        help=" ".join([
            f"Audios with td-idf below this threshold will be ignored.",
            f"Defaults to {default_min_tfidf}."
        ]),
        default=default_min_tfidf
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

    if args.num_audios:
        num_audios = args.num_audios
    else:
        num_audios = get_queries_count()

    min_tfidf = args.min_tfidf
    will_save_graphic = args.save_graphic

    return num_audios, min_tfidf, will_save_graphic


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


def estimate_queries_tfidfs(tfidfs, all_pitch_contour_segmentations, num_audios, min_tfidf, vocabulary):
    queries_tfidfs = {}

    results_mapping = load_expected_results()
    kwargs = {
        "tfidfs": tfidfs,
        # "min_tfidf": min_tfidf,
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
    num_audios, min_tfidf, will_save_graphic = process_args()

    tfidf_data_frame = load_structure(
        structure_name=f'{SONG}_tf_idfs_per_file',
        as_numpy=False,
        as_pandas=True,
        extension='pkl'
    )
    vocabulary = get_vocabulary(tfidf_data_frame)

    all_pitch_contour_segmentations = deserialize_queries_pitch_contour_segmentations(
        num_audios=num_audios
    )

    queries_tfidfs = estimate_queries_tfidfs(
        tfidf_data_frame,
        all_pitch_contour_segmentations,
        num_audios,
        min_tfidf=min_tfidf,
        vocabulary=vocabulary
    )

    data_frame = pd.DataFrame.from_dict(
        queries_tfidfs,
        orient='index',
        dtype=np.float64
    )

    dump_structure(
        data_frame,
        structure_name=f'{QUERY}_tf_idfs_per_file',
        as_numpy=False,
        as_pandas=True,
        extension="pkl"
    )    


if __name__ == "__main__":
    main()
