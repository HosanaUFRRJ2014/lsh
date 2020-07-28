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
    deserialize_queries_pitch_contour_segmentations,
    deserialize_songs_pitch_contour_segmentations,
    dump_structure,
    get_songs_count,
    get_queries_count,
    load_structure
)
from messages import log_invalid_audio_type_error
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
    if args.num_audios:
        num_audios = args.num_audios
    else:
        if audio_type == QUERY:
            num_audios = get_queries_count()
        if audio_type == SONG:
            num_audios = get_songs_count()

    min_tfidf = args.min_tfidf
    will_save_graphic = args.save_graphic

    return num_audios, min_tfidf, audio_type, will_save_graphic


def obtain_song_remaining_pitches(**kwargs):
    tfidfs = kwargs.get("tfidfs")
    filename = kwargs.get("filename")
    original_vector = kwargs.get("original_vector")
    min_tfidf = kwargs.get("min_tfidf")

    # Rebuild the original portion of data of filtered pitches by tfidf
    pitches_and_tfidfs = tfidfs[filename]
    remaining_pitches = []
    for pitch in original_vector:
        if pitches_and_tfidfs[str(pitch)] >= min_tfidf:
            remaining_pitches.append(pitch)

    return remaining_pitches


def obtain_query_remaining_pitches(**kwargs):
    original_vector = kwargs.get("original_vector")
    vocabulary_remaining_pitches = kwargs.get("vocabulary_remaining_pitches")

    remaining_pitches = [
        pitch
        for pitch in original_vector
        if pitch in vocabulary_remaining_pitches
    ]
    return remaining_pitches


def obtain_remaining_pitches(audio_type, **kwargs):
    obtain_remainings = {
        SONG: obtain_song_remaining_pitches,
        QUERY: obtain_query_remaining_pitches
    }

    return obtain_remainings[audio_type](**kwargs)


def extract_remaining_pitches(tfidfs, all_pitch_contour_segmentations, num_audios, min_tfidf, audio_type, vocabulary_remaining_pitches):
    percentages = {}
    no_remaining_pitches_count = 0
    all_remaining_pitches = {}
    kwargs = {
        "tfidfs": tfidfs,
        "min_tfidf": min_tfidf,
        "vocabulary_remaining_pitches": vocabulary_remaining_pitches
    }
    for filename, original_pitch_vector, _onsets, _durations in all_pitch_contour_segmentations:
        # Removes zeros values
        # original_pitch_vector = np.array(original_pitch_vector)
        # original_pitch_vector = original_pitch_vector[np.nonzero(original_pitch_vector)[0]]

        # If query, gets its corresponding song. Otherwise, it's the audio name. 
        # filename = results_mapping.get(filename, filename)
        
        kwargs["filename"] = filename
        kwargs["original_vector"] = original_pitch_vector

        remaining_pitches = obtain_remaining_pitches(audio_type, **kwargs)

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

    return all_remaining_pitches, percentages, no_pitches_percent


def main():
    percentages = {}
    results_mapping = {}
    vocabulary_remaining_pitches = np.array([])
    num_audios, min_tfidf, audio_type, will_save_graphic = process_args()
    tfidfs = load_structure(
        structure_name=f'tf_idfs_per_file',
        as_numpy=False
    )

    if audio_type == SONG:
        all_pitch_contour_segmentations = deserialize_songs_pitch_contour_segmentations(num_audios)
    elif audio_type == QUERY:
        all_pitch_contour_segmentations = deserialize_queries_pitch_contour_segmentations(num_audios)

        # Calculated when this command is ran for songs
        # TODO: Add try/except message
        vocabulary_remaining_pitches = load_structure(
            structure_name=f'{SONG}_remaining_pitches_min_tfidf_{min_tfidf}'
        )

    else:
        log_invalid_audio_type_error(audio_type)
        exit(1)
    

    all_remaining_pitches, percentages, no_pitches_percent = extract_remaining_pitches(
        tfidfs,
        all_pitch_contour_segmentations,
        num_audios,
        min_tfidf=min_tfidf,
        audio_type=audio_type,
        vocabulary_remaining_pitches=vocabulary_remaining_pitches  # empty if audio type is song
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
                "Percentage of remaining pitches after appling TF-IDF",
                f"in {num_audios} {audio_type} (min-tfidf>{min_tfidf})"
            ])
        )

    # Remaining pitches separated by file
    dump_structure(
        structure=all_remaining_pitches,
        structure_name=f'remaining_pitches_min_tfidf_{min_tfidf}_per_{audio_type}'
    )

    if audio_type == SONG:            
        # Set of remaining pitches
        list_remainings = []
        for sublist in all_remaining_pitches.values():
            list_remainings.extend(sublist)
        dump_structure(
            structure=list(set(list_remainings)),
            structure_name=f'{SONG}_remaining_pitches_min_tfidf_{min_tfidf}'
        )
        # dump_structure(
        #     structure=percentages,
        #     structure_name=f'{audio_type}_remaining_pitches_percentages_min_tfidf_{min_tfidf}'
        # )

if __name__ == "__main__":
    main()
