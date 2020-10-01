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
import inflect
from argparse import ArgumentParser
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

from constants import (
    FILES_PATH,
    JACCARD_SIMILARITY,
    COSINE_SIMILARITY,
    RECURSIVE_ALIGNMENT,
    KTRA,
    LINEAR_SCALING,
    BALS,
    MATCHING_ALGORITHMS,
    METRIC_TYPES,
    MAE,
    RMSE
)
from utils import save_graphic
from json_manipulator import (
    dump_structure,
    load_structure
)

# Line color, followed by errobar color
MATCHING_ALGORITHMS_TO_COLORS = {
    JACCARD_SIMILARITY: ("#ffbf00", "#ff6f00"),   # yellow, orange
    COSINE_SIMILARITY: ("#34dbeb", "#00328a") ,   # light blue, dark blue
    RECURSIVE_ALIGNMENT: ("#7bff3d", "#1a5200"),  # light green, dark green  
    KTRA: ("#bb9fd1", "#410073"),                 # light purple, dark purple
    # LINEAR_SCALING: ("#ffe0f3", "#ff039d"),      # light pink, dark pink       
    # BALS: ("#f3e5ab", "#630000")                 # beige, brown
}


def process_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--num_songs",
        "-na",
        type=int,
        help=" ".join([
            "Number of audios to consider. If not informed,",
            "will apply calculations only considering MIREX datasets."
        ])
    )

    parser.add_argument(
        "--matching_algorithm",
        "-ma",
        type=str,
        nargs='+',
        help=f"Options: {MATCHING_ALGORITHMS}."
    )

    parser.add_argument(
        "--min_tfidf",
        "-min",
        type=float,
        nargs='+',
        help=f"Gets the previous calculated  similarities for this minimum TF-IDF."
    )
    args = parser.parse_args()

    num_songs = args.num_songs
    min_tfidfs = args.min_tfidf
    matching_algorithms = args.matching_algorithm

    return num_songs, min_tfidfs, matching_algorithms


def main():
    num_songs, min_tfidfs, matching_algorithms = process_args()

    p = inflect.engine()

    min_tfidfs.sort()

    evaluations_path = "evaluations"
    songs_count_path = f"{num_songs}_songs"
    path = f"{songs_count_path}/{evaluations_path}"
    min_tfidfs = np.array(min_tfidfs)
    fig, ax = plt.subplots()
    plt.xlabel("Minimum TF-IDFs")
    plt.ylabel("Mean Absolute Error (MAE)")
    title = "MAE for pitch extractions below minimum TF-IDFs,\n for {} matching {}".format(
        ", ".join(matching_algorithms),
        p.plural("algorithm", len(matching_algorithms))
    )
    plt.title(title)

    for ytick, matching_algorithm in enumerate(matching_algorithms):
        maes = []
        rmses = []
        for min_tfidf in min_tfidfs:
            mae_filename = f"{path}/{matching_algorithm}_mae_min_tfidf_{min_tfidf}"
            rmse_filename = f"{path}/{matching_algorithm}_rmse_min_tfidf_{min_tfidf}"

            mae = load_structure(
                mae_filename,
                as_numpy=False,
                as_pandas=False,
                extension="txt"
            )
            maes.append(mae)

            rmse = load_structure(
                rmse_filename,
                as_numpy=False,
                as_pandas=False,
                extension="txt"
            )
            rmses.append(rmse)
        
        ax.errorbar(
            x=min_tfidfs,
            y=maes,
            yerr=rmses,
            color=MATCHING_ALGORITHMS_TO_COLORS[matching_algorithm][0],
            ecolor=MATCHING_ALGORITHMS_TO_COLORS[matching_algorithm][1],
            linestyle="dashdot",
            marker='o',
            capsize=5,
            label=matching_algorithm
        )
    
    ax.legend()
    title = title.replace("\n", "")
    plt.savefig(f"{FILES_PATH}/{path}/{title}.png")
    plt.show()


if __name__ == "__main__":
    main()
