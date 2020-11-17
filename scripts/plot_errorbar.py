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
    ABRREV_TO_VERBOSE_NAME,
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
        "--metric",
        "-me",
        type=str,
        choices=METRIC_TYPES
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
    metric = args.metric

    return num_songs, min_tfidfs, matching_algorithms, metric


def main():
    num_songs, min_tfidfs, matching_algorithms, metric = process_args()

    matching_algo_count = len(matching_algorithms)
    p = inflect.engine()

    min_tfidfs.sort()

    evaluations_path = "evaluations"
    songs_count_path = f"{num_songs}_songs"
    data_path = f"{songs_count_path}/{evaluations_path}"
    graphics_path = f"graphics/{songs_count_path}"
    min_tfidfs = np.array(min_tfidfs)
    fig, axs = plt.subplots(
        figsize=[10 * matching_algo_count, 10],
        dpi=80,
        nrows=1,
        ncols=matching_algo_count,
        # sharey='row',
        tight_layout=True  # decreases padding size
    )

    if len(fig.axes) == 1:
        axs = [axs]

    matching_algo_str = ", ".join([
        ABRREV_TO_VERBOSE_NAME.get(m, m)
        for m in matching_algorithms
    ])
    title = "{} for pitch extractions below minimum TF-IDFs\n for {} matching {} ({} songs)".format(
        metric.upper(),
        matching_algo_str,
        p.plural("algorithm", matching_algo_count),
        num_songs
    )
    fig.suptitle(title)

    yticks = []
    ytick_value = 0
    MAX_Y = 15
    for i in range(0, MAX_Y*2+1): 
        yticks.append(ytick_value) 
        ytick_value = ytick_value + 0.5

    yticks = np.array(yticks)

    for i, matching_algorithm in enumerate(matching_algorithms):
        maes = []
        maes_var = []
        maes_std = []
        rmses = []
        for min_tfidf in min_tfidfs:
            mae_filename = f"{data_path}/{matching_algorithm}_mae_min_tfidf_{min_tfidf}"
            rmse_filename = f"{data_path}/{matching_algorithm}_rmse_min_tfidf_{min_tfidf}"
            maes_var_filename = f"{data_path}/{matching_algorithm}_mae_var_min_tfidf_{min_tfidf}"

            mae, mae_std, mae_var = load_structure(
                mae_filename
            )
            maes.append(mae)
            maes_var.append(mae_var)
            maes_std.append(mae_std)

            rmse = load_structure(
                rmse_filename
            )
            rmses.append(rmse)
        

        if metric == MAE:
            axs[i].errorbar(
                x=min_tfidfs,
                y=maes,
                # yerr=maes_std,
                color=MATCHING_ALGORITHMS_TO_COLORS[matching_algorithm][0],
                linestyle="dashdot",
                marker='o',
                capsize=7,
                label=f"{matching_algorithm} - MAE"
            )

        if metric == RMSE:
            axs[i].errorbar(
                x=min_tfidfs,
                y=rmses,
                color=MATCHING_ALGORITHMS_TO_COLORS[matching_algorithm][1],
                ecolor=MATCHING_ALGORITHMS_TO_COLORS[matching_algorithm][1],
                linestyle="dashdot",
                marker='^',
                capsize=7,
                label=f"{matching_algorithm} - RMSE",
            )
        axs[i].set_ylim(
            bottom=0
        )
        axs[i].set_title(ABRREV_TO_VERBOSE_NAME[matching_algorithm])
        axs[i].legend()
        axs[i].set_xticks(min_tfidfs)
        axs[i].set_yticks(yticks)
        axs[i].tick_params(
            axis='x',  # apply only for x-axis
            labelrotation=35,
            direction='inout',
            length=5
        )
        axs[i].set_xlabel("Minimum TF-IDFs")


    title = title.replace('\n', '')
    plt.savefig(f"{graphics_path}/{title}.png")
    # plt.show()


if __name__ == "__main__":
    main()
