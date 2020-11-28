# Includes the parent directory into sys.path, to make imports work
import os, os.path, sys
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
from itertools import cycle
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
from messages import log_invalid_axes_number_error

NAME_KEY = "name"
LINE_KEY = "line"
LINE_STYLE_KEY = "style"
LINE_MARKER_KEY = "marker"
LINE_COLOR_KEY = "color"
LINE_MAE_COLOR_KEY = f"{MAE}Color"
LINE_RMSE_COLOR_KEY = f"{RMSE}Color"


MATCHING_ALGORITHMS_META_INFO = {
    JACCARD_SIMILARITY: {
        NAME_KEY: ABRREV_TO_VERBOSE_NAME[JACCARD_SIMILARITY],
        LINE_KEY: {
            LINE_STYLE_KEY: "dotted",
            LINE_MARKER_KEY: "o",
            LINE_COLOR_KEY: "#00a8ab",  # light blue
            LINE_MAE_COLOR_KEY: "#00a8ab",  # light blue 
            LINE_RMSE_COLOR_KEY: "#00308f"  # dark blue
            
        }

    },
    COSINE_SIMILARITY: {
        NAME_KEY: ABRREV_TO_VERBOSE_NAME[COSINE_SIMILARITY],
        LINE_KEY: {
            LINE_STYLE_KEY: "dashdot",
            LINE_MARKER_KEY: "v",
            LINE_COLOR_KEY: "#09ff00", # light green
            LINE_MAE_COLOR_KEY: "#09ff00", # light green 
            LINE_RMSE_COLOR_KEY: "#163800"  # dark green
        }
    },
    RECURSIVE_ALIGNMENT: {
        NAME_KEY: ABRREV_TO_VERBOSE_NAME[RECURSIVE_ALIGNMENT],
        LINE_KEY: {
            LINE_STYLE_KEY: "dotted",
            LINE_MARKER_KEY: "X",
            LINE_COLOR_KEY: "#e01709",  # red
            LINE_MAE_COLOR_KEY: "#e01709",  # red
            LINE_RMSE_COLOR_KEY: "#de8900"  # orange
        }
    },
    KTRA: {
        NAME_KEY: ABRREV_TO_VERBOSE_NAME[KTRA],
        LINE_KEY: {
            LINE_STYLE_KEY: "dotted",
            LINE_MARKER_KEY: "s",
            LINE_COLOR_KEY: "#6726ff",  # light purple 
            LINE_MAE_COLOR_KEY: "#6726ff",  # light purple 
            LINE_RMSE_COLOR_KEY: "#1a0845"  # dark purple
        }

    },
    LINEAR_SCALING: {  
        NAME_KEY: ABRREV_TO_VERBOSE_NAME[LINEAR_SCALING],
        LINE_KEY: {
            LINE_STYLE_KEY: "dotted",
            LINE_MARKER_KEY: "d",
            LINE_COLOR_KEY: "#ffe0f3",  # light pink
            LINE_MAE_COLOR_KEY: "#ffe0f3",  # light pink 
            LINE_RMSE_COLOR_KEY: "#ff039d"  # dark pink
        }
    },
    BALS: {
        NAME_KEY: ABRREV_TO_VERBOSE_NAME[BALS],
        LINE_KEY: {
            LINE_STYLE_KEY: "dotted",
            LINE_MARKER_KEY: "*",
            LINE_COLOR_KEY: "#f3e5ab",  # beige
            LINE_MAE_COLOR_KEY: "#f3e5ab",  # beige
            LINE_RMSE_COLOR_KEY: "#630000"  # brown
        }
    }

}


def process_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--num_songs",
        "-num_songs",
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

    parser.add_argument(
        "--num_axes",
        "-na",
        type=int,
        help=" ".join([
            "Number of axes to consider. If not informed,",
            "will use only one. It must be one or the number of informed",
            "matching algorithms."
        ]),
        default=1
    )

    args = parser.parse_args()

    num_songs = args.num_songs
    min_tfidfs = args.min_tfidf
    matching_algorithms = args.matching_algorithm
    metric = args.metric
    num_axes = args.num_axes

    return num_songs, min_tfidfs, matching_algorithms, metric, num_axes


def create_graphics_folder(graphics_path):
    os.makedirs(graphics_path, exist_ok=True)


def make_title(matching_algorithms, metric, num_songs):
    p = inflect.engine()

    matching_algorithm_count = len(matching_algorithms)
    matching_algo_str = ""
    matching_algo_str = ", \n".join([
        ABRREV_TO_VERBOSE_NAME.get(m, m)
        for m in matching_algorithms[:-1]
    ])

    if matching_algorithm_count > 1:
        matching_algo_str += "\n and "

    last_or_unique_ma = matching_algorithms[-1]
    matching_algo_str += ABRREV_TO_VERBOSE_NAME.get(
        last_or_unique_ma,
        last_or_unique_ma
    )

    title = "{} for pitch extractions below minimum TF-IDFs for\n {}\n matching {} ({} songs)".format(
        metric.upper(),
        matching_algo_str,
        p.plural("algorithm", matching_algorithm_count),
        num_songs
    )

    return title


def normalize_axes(figure, axes):
    axes_count = len(figure.axes)
    if axes_count == 1:
        axes = [axes]

    return axes


def make_yticks(metric):
    yticks = []
    ytick_value = 0

    if metric == MAE:
        MAX_Y = 2*10+1
        step = 0.1
    elif metric == RMSE:
        MAX_Y = 15*2+1
        step = 0.5

    for i in range(0, MAX_Y): 
        yticks.append(ytick_value) 
        ytick_value = ytick_value + step

    yticks = np.array(yticks)

    return yticks


def validate_axes_count(axes_count, matching_algorithm_count):
    has_valid_multiple_axes = axes_count == matching_algorithm_count
    is_single_axe = axes_count == 1
    
    axes_are_valid = has_valid_multiple_axes or is_single_axe

    if not axes_are_valid:
        log_invalid_axes_number_error(axes_count, matching_algorithm_count)
        exit(1)

    
def main():
    num_songs, min_tfidfs, matching_algorithms, metric, num_axes = process_args()

    matching_algorithm_count = len(matching_algorithms)
    min_tfidfs.sort()

    evaluations_path = "evaluations"
    songs_count_path = f"{num_songs}_songs"
    data_path = f"{songs_count_path}/{evaluations_path}"
    graphics_path = f"graphics/{songs_count_path}"

    create_graphics_folder(graphics_path)

    min_tfidfs = np.array(min_tfidfs)

    figure_width = 10 if num_axes == 1 else 10 * matching_algorithm_count


    if num_axes == 1:
        figure_width = 10
        dpi = 100
    else:
        figure_width = 10 * matching_algorithm_count
        dpi = 80

    figure, axes = plt.subplots(
        figsize=[figure_width, 10],
        dpi=dpi,
        nrows=1,
        ncols=num_axes,
        tight_layout=True  # decreases padding size
    )

    yticks = make_yticks(metric=metric)
    axes = normalize_axes(figure=figure, axes=axes)
    axes_count = len(axes)

    # If not valid, exits program
    validate_axes_count(axes_count, matching_algorithm_count)

    title = make_title(
        matching_algorithms=matching_algorithms,
        metric=metric,
        num_songs=num_songs
    )
    figure.suptitle(title)


    markersize = 8
    capsize = 25
    linewidth = 3
 
    for _axes_count, matching_algorithm in zip(cycle([axes_count]), matching_algorithms):
        i = _axes_count - 1
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

        matching_algorithm_name = MATCHING_ALGORITHMS_META_INFO[matching_algorithm][NAME_KEY]
        linestyle = MATCHING_ALGORITHMS_META_INFO[matching_algorithm][LINE_KEY][LINE_STYLE_KEY]
        marker = MATCHING_ALGORITHMS_META_INFO[matching_algorithm][LINE_KEY][LINE_MARKER_KEY]
        color = MATCHING_ALGORITHMS_META_INFO[matching_algorithm][LINE_KEY][LINE_COLOR_KEY]
        
        if metric == MAE:
            # mae_color = MATCHING_ALGORITHMS_META_INFO[matching_algorithm][LINE_KEY][LINE_MAE_COLOR_KEY]
            axes[i].errorbar(
                x=min_tfidfs,
                y=maes,
                # yerr=maes_std,
                color=color,
                alpha=0.7,
                linestyle=linestyle,
                marker=marker,
                markersize=markersize,
                capsize=capsize,
                linewidth=linewidth,
                label=f"{matching_algorithm_name}"
            )

        if metric == RMSE:
            # rmse_color = MATCHING_ALGORITHMS_META_INFO[matching_algorithm][LINE_KEY][LINE_RMSE_COLOR_KEY]
            axes[i].errorbar(
                x=min_tfidfs,
                y=rmses,
                color=color,
                alpha=0.7,
                ecolor=color,
                linestyle=linestyle,
                marker=marker,
                markersize=markersize,
                capsize=capsize,
                linewidth=linewidth,
                label=f"{matching_algorithm_name}",
            )
        axes[i].set_ylim(
            bottom=0
        )

        if axes_count != 1:
            axes[i].set_title(ABRREV_TO_VERBOSE_NAME[matching_algorithm])
        axes[i].legend()
        axes[i].set_xticks(min_tfidfs)
        axes[i].set_yticks(yticks)
        axes[i].tick_params(
            axis='x',  # apply only for x-axis
            labelrotation=35,
            direction='inout',
            length=10
        )
        axes[i].set_xlabel("Minimum TF-IDFs")


    # Workaround to remove y labels from second graphic onwards
    # if matching_algorithm_count > 1:
    #     for i in range(1, len(axes)):
    #         axes[i].tick_params(
    #             axis='y',
    #             labelcolor='white'
    #         )

    title = title.replace('\n', '')
    plt.savefig(f"{graphics_path}/{title}.png")
    # plt.show()


if __name__ == "__main__":
    main()
