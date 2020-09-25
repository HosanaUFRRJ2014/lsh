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
import textwrap
import subprocess

from constants import (
    SONG,
    QUERY,
    SIMILARITY_MATCHING_ALGORITHMS,
    ALIGNMENT_MATCHING_ALGORITHMS,
    METRIC_TYPES
)
from loader import get_songs_count
from json_manipulator import load_structure

def process_args():
    description = textwrap.dedent('''\
    Executes all of the TF-IDF process.
    ''')

    default_num_songs = get_songs_count()
    default_min_tfidf = 0.01

    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description=description
    )

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
        nargs='+',
        type=float,
        help=" ".join([
            f"Audios with td-idf below this threshold will be ignored.",
            f"Defaults to {default_min_tfidf}."
        ]),
        default=[default_min_tfidf]
    )


    args = parser.parse_args()
    num_songs = args.num_songs
    min_tfidfs = args.min_tfidf

    return num_songs, min_tfidfs


def will_exec_command(structure_name):
    try:
        structure = load_structure(structure_name, as_numpy=False, as_pandas=True, extension="pkl")
        will_exec = False
    except FileNotFoundError:
        will_exec = True

    return will_exec


def main():
    commands = []
    # matching_algorithms = SIMILARITY_MATCHING_ALGORITHMS + ALIGNMENT_MATCHING_ALGORITHMS
    matching_algorithms = ALIGNMENT_MATCHING_ALGORITHMS
    num_songs, min_tfidfs = process_args()

    # will_exec_tfidf = will_exec_command(f"{num_songs}_songs/{SONG}_tf_idfs_per_file")
    #
    # if will_exec_tfidf:
    #     commands.extend([
    #         f"python scripts/song_tfidf_calculation.py --num_songs {num_songs}",
    #         f"python scripts/query_tfidf_calculation.py --num_songs {num_songs}"
    #     ])

    for min_tfidf in min_tfidfs:
        # commands.extend([
        #     f"python scripts/tfidf_pitch_extraction.py --audio_type song --min_tfidf {min_tfidf} --num_songs {num_songs}",
        #     f"python scripts/tfidf_pitch_extraction.py --audio_type query --min_tfidf {min_tfidf} --num_songs {num_songs}"
        # ])

        for matching_algorithm in matching_algorithms:
            commands.append(
                f"python scripts/calculate_similarities.py --min_tfidf {min_tfidf} -ma {matching_algorithm}  --num_songs {num_songs}"
            )

            will_exec_sim_for_originals = will_exec_command(
                structure_name=f"{num_songs}_songs/similarities/{matching_algorithm}"
            )
            if will_exec_sim_for_originals:
                commands.append(
                    f"python scripts/calculate_similarities.py -ma {matching_algorithm}  --num_songs {num_songs}"
                )

            for metric_type in METRIC_TYPES:
                commands.extend([
                    f"python scripts/evaluation_metrics.py --metric {metric_type} --min_tfidf {min_tfidf} -ma {matching_algorithm}  --num_songs {num_songs}"
                ])

    steps_count = len(commands)
    for step, command in enumerate(commands, 1):
        print(f"Step {step} / {steps_count}")
        print(command)
        returned = subprocess.run(command, shell=True)
        if returned.returncode != 0:
            print(command)
            exit(1)


if __name__ == '__main__':
    main()
