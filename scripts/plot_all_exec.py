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
import subprocess

from constants import (
    SIMILARITY_MATCHING_ALGORITHMS,
    ALIGNMENT_MATCHING_ALGORITHMS,
    METRIC_TYPES
)

def main():
    commands = []
    songs_count = [
        "346",
        "1000",
        "2000",
        "3000",
        "5000",
        "10000",
        "25000",
        "30000",
        "50000"
    ]

    matching_algorithms = SIMILARITY_MATCHING_ALGORITHMS + ALIGNMENT_MATCHING_ALGORITHMS

    
    for num_songs in songs_count:
        for matching in matching_algorithms:
            for metric in METRIC_TYPES:
                commands.append(
                    f"python scripts/plot_errorbar.py --num_songs {num_songs} --min_tfidf 0.01 0.001 0.0001 -ma {matching} -me {metric}"
                )

    
    steps_count = len(commands)
    for step, command in enumerate(commands, 1):
        print(
            f"Step {step} / {steps_count};", 
            "{} % done.".format(round((step * 100)/steps_count, ndigits=2))
        )
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()