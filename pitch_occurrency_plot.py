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

array_of_pitch_values = []

# Number of audios to consider in the grafic
num_audios = 300
min_tfidf = 0.1
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
    # FIXME: tem música ficando sem pitch significativo
    for li in list(tfidfs.values()):
        values.extend(li)
    tfidfs_values = np.array(values)
    histogram, bins, patches = plt.hist(
        x=tfidfs_values,
        bins='auto',
        histtype='stepfilled',
        color='#0504aa',
        alpha=0.7,
        rwidth=0.85
    )
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Pitch Value')
    plt.ylabel('Frequency')
    plt.title(
        f'TF-IDF of pitch values in a sample of {num_audios} songs (min-tfidf>{min_tfidf})'
    )

    max_frequency = histogram.max()

    # Set a clean upper y-axis limit.
    if max_frequency % 10:
        y_max = np.ceil(max_frequency / 10) * 10
    else:
        y_max = max_frequency + 10

    plt.ylim(ymax=y_max)
    plt.show()