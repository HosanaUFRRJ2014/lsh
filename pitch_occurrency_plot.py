'''
python essentia_examples.py
'''

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from essentia.standard import (
    MusicExtractor,
    EqloudLoader,
    PitchMelodia
)
from math import floor

from json_manipulator import deserialize_songs_pitch_contour_segmentations, dump_structure
from utils import calculate_tfidfs


def plot_graphic(values, xlabel, ylabel, title):
    values_as_nparray = np.array(values)
    histogram, bins, patches = plt.hist(
        x=values_as_nparray,
        bins='auto',
        histtype='stepfilled',
        color='#0504aa',
        alpha=0.7,
        rwidth=0.85
    )
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    max_frequency = histogram.max()

    # Set a clean upper y-axis limit.
    if max_frequency % 10:
        y_max = np.ceil(max_frequency / 10) * 10
    else:
        y_max = max_frequency + 10

    plt.ylim(ymax=y_max)
    plt.show()


def extract_plotable_tfidfs(tfidfs, all_pitch_contour_segmentations):
    values = []

    for filename, original_pitch_vector, _, _ in all_pitch_contour_segmentations:
        remaining_pitches_and_tfidfs = tfidfs[filename]

        # ignore audios without remainings
        if len(remaining_pitches_and_tfidfs) == 0:
            continue

        _tfidfs, _ = zip(*remaining_pitches_and_tfidfs)     
        values.extend(_tfidfs)

    return values


def calculate_remaining_pitches_percents(tfidfs, all_pitch_contour_segmentations, num_audios):
    percentages = {}
    no_remaining_pitches_count = 0
    for filename, original_pitch_vector, _, _ in all_pitch_contour_segmentations:
        # Removes zeros values
        original_pitch_vector = np.array(original_pitch_vector)
        original_pitch_vector = original_pitch_vector[np.nonzero(original_pitch_vector)[0]]

        remaining_pitches_and_tfidfs = tfidfs[filename]

        # ignore audios without remainings
        if len(remaining_pitches_and_tfidfs) == 0:
            no_remaining_pitches_count += 1
            continue

        _, pitches_count = zip(*remaining_pitches_and_tfidfs)

        # Rebuild the original portion of data of filtered pitches by tfidf
        remaining_portion_of_pitches = []
        for pitch, count in pitches_count:
            remaining_portion_of_pitches.extend(
                [pitch for i in range(count)]
            )

        # calculate percentage of remainings
        remaining_length = len(remaining_portion_of_pitches)
        original_length = len(original_pitch_vector)

        percent = (remaining_length/original_length) * 100
        percentages[filename] = percent

        print("Original size:", original_length)
        print("Remaining size:", remaining_length)
        print("Percent:", percent, " %", "\n")

    no_pitches_percent = (no_remaining_pitches_count/num_audios) * 100

    return percentages, no_pitches_percent


def process_args():
    parser = ArgumentParser()

    default_num_audios = 300
    default_min_tfidf = 0.01
    default_save_in_file = False
    default_plot_tfdfs = False
    default_plot_remaining_percents = True

    parser.add_argument(
        "--num_audios",
        "-na",
        type=float,
        help=f"Number of audios to consider. Defaults to {default_num_audios}.",
        default=default_num_audios
    )

    parser.add_argument(
        "--min_tfidf",
        "-min",
        type=float,
        help=f"Audios with td-idf below this threshold will be ignored. Defaults to {default_min_tfidf}.",
        default=default_min_tfidf
    )

    parser.add_argument(
        "--save",
        type=bool,
        help=f"If True, saves tf-idfs calculations in a file. Defaults to {default_save_in_file}.",
        default=default_save_in_file
    )


    parser.add_argument(
        "--plot_tfidfs",
        type=bool,
        help=f"If True, plots the tfidfs graphic. Defaults to {default_plot_tfdfs}.",
        default=default_plot_tfdfs
    )

    parser.add_argument(
        "--plot_remaining_pitches_percentage",
        "-plot_rpp",
        type=bool,
        help=f"If True, plots the remaining pitches percentage graphic. Defaults to {default_plot_remaining_percents}.",
        default=default_plot_remaining_percents
    )

    args = parser.parse_args()

    num_audios = args.num_audios
    min_tfidf = args.min_tfidf
    save_in_file = args.save
    plot_tfidfs = args.plot_tfidfs
    plot_rpp = args.plot_remaining_pitches_percentage


    return num_audios, min_tfidf, save_in_file, plot_tfidfs, plot_rpp


def main():
    """
    python pitch_occurrency_plot.py --min_tfidf MIN_TFIDF 
    """
    percentages = {}
    num_audios, min_tfidf, save_in_file, plot_tfidfs, plot_rpp = process_args() 

    all_pitch_contour_segmentations = deserialize_songs_pitch_contour_segmentations(num_audios)
    tfidfs = calculate_tfidfs(
        num_audios, all_pitch_contour_segmentations, min_tfidf=min_tfidf
    )

    if plot_tfidfs:
        values = extract_plotable_tfidfs(tfidfs, all_pitch_contour_segmentations)
        plot_graphic(
            values,
            xlabel='TF-IDFS',
            ylabel='Amount of audios',
            title=f'Appling TF-IDF in {num_audios} songs (min-tfidf>{min_tfidf})'
        )


    if plot_rpp:
        # rpp = remaining pitches percentage
        percentages, no_pitches_percent = calculate_remaining_pitches_percents(
            tfidfs, all_pitch_contour_segmentations, num_audios
        )

        print(
            " ".join([
                f"Percentage of audios without pitches for {num_audios}",
                f"audios: {no_pitches_percent} %",
                f"(min-tfidf>{min_tfidf})"
            ])
        )

        plot_graphic(
            list(percentages.values()),
            xlabel='Percentage',
            ylabel='Amount of audios',
            title=f'Percentage of remaining pitches after appling TF-IDF in {num_audios} songs (min-tfidf>{min_tfidf})'
        )


    if save_in_file:
        dump_structure(structure=tfidfs, structure_name='tf_idfs')

        if percentages:
            dump_structure(
                structure=percentages,
                structure_name='remaining_pitches_percentages'
            )


if __name__ == "__main__":
    main()
