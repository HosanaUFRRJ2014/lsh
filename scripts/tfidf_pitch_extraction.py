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
from argparse import (
    ArgumentParser,
    RawDescriptionHelpFormatter
)
from math import floor, log2
import textwrap

from essentia.standard import (
    MusicExtractor,
    EqloudLoader,
    PitchMelodia
)
import numpy as np

from constants import (
    AUDIO_TYPES,
    QUERY,
    SONG
)
from json_manipulator import (
    deserialize_audios_pitch_contour_segmentations,
    dump_structure,
    get_songs_count,
    get_queries_count,
    load_structure
)
from messages import ( 
   log_forgotten_step_warn
)
from utils import save_graphic


def process_args():
    description = textwrap.dedent('''\
    Extracts pitches above the min_tfidf threshold.
    Saves:
        - Remaining pitches separated by file (songs and queries filenames),
        - Set of remaining pitches,
        - etc
    ''')
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description=description
    )

    default_min_tfidf = 0.01
    default_save_graphic = True
    default_num_songs = get_songs_count()

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
        "--audio_type",
        "-type",
        type=str,
        help="Options: {}. Defaults to {}".format(
            ', '.join(AUDIO_TYPES),
            SONG
        ),
        default=SONG
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

    audio_type = args.audio_type
    num_songs = args.num_songs

    min_tfidf = args.min_tfidf
    will_save_graphic = args.save_graphic

    return num_songs, min_tfidf, audio_type, will_save_graphic


def obtain_remaining_pitches(**kwargs):
    tfidfs = kwargs.get("tfidfs")
    filename = kwargs.get("filename")
    original_vector = kwargs.get("original_vector")
    min_tfidf = kwargs.get("min_tfidf")

    remaining_pitches = []
    
    # Rebuild the original portion of data of filtered pitches by tfidf
    try:
        pitches_and_tfidfs = tfidfs.loc[filename]
    except:
        print(f"filename {filename} has no tfidf value. Skipping")
        return remaining_pitches
        
    for pitch in original_vector:
        tfidf = pitches_and_tfidfs.get(pitch)
        if tfidf and tfidf > min_tfidf:
            remaining_pitches.append(pitch)

    return remaining_pitches


def extract_remaining_pitches(
    tfidfs, all_pitch_contour_segmentations, num_audios, min_tfidf
):
    percentages = {}
    no_remaining_pitches_count = 0
    all_remaining_pitches = {}

    kwargs = {
        "tfidfs": tfidfs,
        "min_tfidf": min_tfidf
    }
    for filename, original_pitch_vector, _onsets, _durations in all_pitch_contour_segmentations:
        # Removes zeros values
        # original_pitch_vector = np.array(original_pitch_vector)
        # original_pitch_vector = original_pitch_vector[np.nonzero(original_pitch_vector)[0]]

        # If query, gets its corresponding song. Otherwise, it's the audio name. 
        # filename = results_mapping.get(filename, filename)
        
        kwargs["filename"] = filename
        kwargs["original_vector"] = original_pitch_vector
        
        remaining_pitches = obtain_remaining_pitches(**kwargs)

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

        #print("Original size:", original_length)
        #print("Remaining size:", remaining_length)
        #print("Percent:", percent, " %", "\n")

    no_pitches_percent = (no_remaining_pitches_count/num_audios) * 100

    return all_remaining_pitches, percentages, no_pitches_percent


def main():
    num_songs, min_tfidf, audio_type, will_save_graphic = process_args()

    if audio_type == QUERY:
        num_audios = get_queries_count()
    else:
        num_audios = num_songs

    try:
        all_pitch_contour_segmentations = deserialize_audios_pitch_contour_segmentations(audio_type, num_audios)
        tfidfs = load_structure(
            structure_name=f'{num_songs}_songs/{audio_type}_tf_idfs_per_file',
            as_numpy=False,
            as_pandas=True,
            extension="pkl"
        )
    except KeyError:
        # raised by deserialize audios function
        exit(1)
    except FileNotFoundError as error:
        log_forgotten_step_warn(error, audio_type)
        exit(1)

    all_remaining_pitches, percentages, no_pitches_percent = extract_remaining_pitches(
        tfidfs,
        all_pitch_contour_segmentations,
        num_audios,
        min_tfidf=min_tfidf
    )
    print(
        " ".join([
            f"Percentage of audios without pitches for {num_audios}",
            f"audios: {no_pitches_percent} %",
            f"(min-tfidf>{min_tfidf})"
        ])
    )

    if will_save_graphic:
        save_graphic(
            list(percentages.values()),
            xlabel='Percentage',
            ylabel='Amount of audios',
            title=" ".join([
                f"{num_songs}_songs/Percentage of remaining pitches after appling TF-IDF",
                f"in {num_audios} {audio_type}s (min-tfidf>{min_tfidf})"
            ])
        )

    # Remaining pitches separated by file
    dump_structure(
        structure=all_remaining_pitches,
        structure_name=f"{num_songs}_songs/remaining_pitches_min_tfidf_{min_tfidf}_per_{audio_type}"
    )


if __name__ == "__main__":
    main()
