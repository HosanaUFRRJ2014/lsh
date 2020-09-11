# Includes the parent directory into sys.path, to make imports work
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
    get_songs_count
)
from utils import save_graphic


def process_args():
    description = textwrap.dedent('''\
    Calculates tfidf of all of the vocabulary.
    ''')
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description=description
    )

    default_num_audios = get_songs_count()
    default_min_tfidf = 0.01
    default_plot_tfdfs = False
    default_calc_remaining_percents = True

    parser.add_argument(
        "--num_songs",
        "-na",
        type=int,
        help=" ".join([
            "Number of audios to consider. If not informed,",
            "will apply calculations only considering MIREX datasets."
        ]),
        default=default_num_audios
    )

    parser.add_argument(
        "--save_tfidfs_graphic",
        type=bool,
        help=" ".join([
            "If True, calculates and saves the tfidfs and its graphic."
            f"Defaults to {default_plot_tfdfs}."
        ]),
        default=default_plot_tfdfs
    )

    args = parser.parse_args()

    num_audios = args.num_audios
    save_tfidfs_graphic = args.save_tfidfs_graphic

    return num_audios, save_tfidfs_graphic


def calculate_pitches_counts(pitch_values):
    """
    Inspired in Term Frequency (TF). Determinates, for each pitch p in a
    pitch array arr, the number of occurencies of p in arr. This number of
    occurrencies is divided by the total number of pitches in arr.

    See more details at:
    https://mungingdata.wordpress.com/2017/11/25/episode-1-using-tf-idf-to-identify-the-signal-from-the-noise/
    """
    unique_elements, counts = np.unique(
        pitch_values, return_counts=True
    )

    pitches_counts = {}

    for pitch, count in zip(unique_elements, counts):
        tf = count/len(pitch_values)
        pitches_counts[pitch] = tf

    return pitches_counts


def calculate_inversed_pitches_values_occurrencies(
    num_audios, array_of_pitch_values
):
    """Inspired in Inverse-Document-Frequency (IDF).
    Number of the songs in the dataset divided by the number of the docs in
    which a certain pitch appears.
    See more at:
    https://mungingdata.wordpress.com/2017/11/25/episode-1-using-tf-idf-to-identify-the-signal-from-the-noise/

    """
    inversed_occurrencies = {}
    num_audios_with_pitch = {}
    for pitches_values in array_of_pitch_values:
        unique_pitches = np.unique(pitches_values)
        for pitch in unique_pitches:
            if pitch in num_audios_with_pitch:
                num_audios_with_pitch[pitch] += 1
            else:
                num_audios_with_pitch[pitch] = 1

    for pitch, _num_appearences in num_audios_with_pitch.items():
        inversed_occurrency = log2(
            num_audios/num_audios_with_pitch[pitch]
        )
        inversed_occurrencies[pitch] = inversed_occurrency

    return inversed_occurrencies


def calculate_tfidfs(num_audios, all_pitch_contour_segmentations):
    """Inspired in Term-Frequency, Inverse-Document-Frequency (TFIDF).

    See more at:
    https://mungingdata.wordpress.com/2017/11/25/episode-1-using-tf-idf-to-identify-the-signal-from-the-noise/
    """
    filenames = []
    array_of_pitch_values = []
    for filename, pitch_values, _onset, _duration in all_pitch_contour_segmentations:
        pitch_values = np.array(pitch_values)
        # Removes zeros values
        # pitch_values = pitch_values[np.nonzero(pitch_values)[0]]
        filenames.append(filename)
        array_of_pitch_values.append(pitch_values)

    idfs = calculate_inversed_pitches_values_occurrencies(
        num_audios,
        array_of_pitch_values
    )

    tfs_of_all_audios = []
    for pitch_values in array_of_pitch_values:
        audio_tfs = calculate_pitches_counts(pitch_values)
        tfs_of_all_audios.append(audio_tfs)

    # maps tf-idfs of all pitches in an audio for each audio
    tfidfs_per_audios = {}
    for filename, audio_tfs in zip(filenames, tfs_of_all_audios):
        tfidf_audio = {}
        for pitch, pitch_tf in audio_tfs.items():
            idf = idfs[pitch]
            tfidf = pitch_tf * idf
            tfidf_audio[pitch] = tfidf

        tfidfs_per_audios[filename] = tfidf_audio

    return tfidfs_per_audios


def extract_plotable_tfidfs(tfidfs, all_pitch_contour_segmentations, min_tfidf=0):
    values = []

    for filename, _original_pitch_vector, _, _ in all_pitch_contour_segmentations:
        remaining_pitches_and_tfidfs = tfidfs[filename]

        # ignore audios without remainings
        if len(remaining_pitches_and_tfidfs) == 0:
            continue

        _tfidfs, _ = zip(*remaining_pitches_and_tfidfs)

        if _tfidfs >= min_tfidf:
            values.extend(_tfidfs)

    return values


def main():
    percentages = {}
    num_songs, save_tfidfs_graphic = process_args()

    all_pitch_contour_segmentations = deserialize_songs_pitch_contour_segmentations(num_songs)
    tfidfs = calculate_tfidfs(
        num_songs, all_pitch_contour_segmentations
    )

    # if save_tfidfs_graphic:
    #     values = extract_plotable_tfidfs(
    #         tfidfs,
    #         all_pitch_contour_segmentations,
    #         min_tfidf=min_tfidf
    #     )
    #     save_graphic(
    #         values,
    #         xlabel='TF-IDFS',
    #         ylabel='Amount of audios',
    #         title=f'TF-IDF in {num_audios} songs (min-tfidf>{min_tfidf})'
    #     )

    data_frame = pd.DataFrame.from_dict(tfidfs, orient='index', dtype=np.float64)

    dump_structure(
        data_frame,
        structure_name=f"{num_songs}_songs/{SONG}_tf_idfs_per_file",
        as_numpy=False,
        as_pandas=True,
        extension="pkl"
    )

    # dump_structure(structure=data_frame, structure_name=f'{SONG}_tf_idfs_per_file')


if __name__ == "__main__":
    main()
