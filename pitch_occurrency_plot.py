'''
python essentia_examples.py
'''

import numpy as np
import matplotlib.pyplot as plt
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




###############################################
array_of_pitch_values = []

# Number of audios to consider in the grafic
num_audios = 300
min_tfidf = 0.0001
save_in_file = False
plot = True

all_pitch_contour_segmentations = deserialize_songs_pitch_contour_segmentations(
    num_audios
)

tfidfs = calculate_tfidfs(num_audios, all_pitch_contour_segmentations, min_tfidf=min_tfidf)

if save_in_file:
    dump_structure(structure=tfidfs, structure_name='tf_idfs')


if plot:
    values = []
    percentages = []
    for filename, original_pitch_vector, _, _ in all_pitch_contour_segmentations:
        # Removes zeros values
        original_pitch_vector = np.array(original_pitch_vector)
        original_pitch_vector = original_pitch_vector[np.nonzero(original_pitch_vector)[0]]

        remaining_pitches_and_tfidfs = tfidfs[filename]

        _tfidfs, pitches_count = zip(*remaining_pitches_and_tfidfs)
        remaining_portion_of_pitches = []
        
        values.extend(_tfidfs)

        # Rebuild the original portion of data of filtered pitches by tfidf
        for pitch, count in pitches_count:
            remaining_portion_of_pitches.extend(
                [pitch for i in range(count)]
            )

        remaining_length = len(remaining_portion_of_pitches)
        original_length = len(original_pitch_vector)

        percent = (remaining_length/original_length) * 100
        percentages.append(percent)
        # print("Original set:", set(original_pitch_vector))

        print("Original size:", original_length)

        # print("Remaining portion:", [p for p, c in pitches_count])
        print("Remaining size:", remaining_length)
        
        print("Percent:", percent, " %", "\n")

    # plot_graphic(
    #     values,
    #     xlabel='TF-IDFS',
    #     ylabel='Amount of audios',
    #     title=f'Appling TF-IDF in {num_audios} songs (min-tfidf>{min_tfidf})'
    # )

    plot_graphic(
        percentages,
        xlabel='Percentage',
        ylabel='Amount of audios',
        title=f'Percentage of remaining pitches after appling TF-IDF in {num_audios} songs (min-tfidf>{min_tfidf})'
    )
