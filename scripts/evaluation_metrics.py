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
from argparse import ArgumentParser
from math import sqrt
import numpy as np

from constants import (
    JACCARD_SIMILARITY,
    MATCHING_ALGORITHMS,
    METRIC_TYPES,
    MAE,
    RMSE
)
from json_manipulator import (
    dump_structure,
    load_structure
)


def process_args():
    parser = ArgumentParser()

    default_num_songs = None # get_songs_count() + get_expanded_songs_count()


    parser.add_argument(
        "--num_songs",
        "-na",
        type=int,
        help=" ".join([
            "Number of audios to consider. If not informed,",
            "will apply calculations only considering MIREX datasets."
        ]),
        default=default_num_songs
    )

    parser.add_argument(
        "--metric_type",
        "-me",
        type=str,
        help=f"Metric to be applied. Options: {METRIC_TYPES}"
    )

    parser.add_argument(
        "--matching_algorithm",
        "-ma",
        type=str,
        help="Options: {}. Defaults to {}".format(
            ', '.join(MATCHING_ALGORITHMS),
            JACCARD_SIMILARITY
        ),
        default=JACCARD_SIMILARITY,
        choices=MATCHING_ALGORITHMS
    )

    parser.add_argument(
        "--min_tfidf",
        "-min",
        type=float,
        help=f"Gets the previous calculated  similarities for this minimum TF-IDF."
    )
    args = parser.parse_args()

    num_songs = args.num_songs
    min_tfidf = args.min_tfidf
    metric_type = args.metric_type
    matching_algorithm = args.matching_algorithm

    return num_songs, min_tfidf, metric_type, matching_algorithm


def mean_absolute_error(similarities, tfidf_similarities, square=False):
    """
    Mean Absolute Error (MAE).
    Calculates MAE between similarities and similarities after aplication of TF-IDF.
    """
    if not isinstance(similarities, np.ndarray):
        similarities = np.array(similarities)
    
    if not isinstance(tfidf_similarities, np.ndarray):
        tfidf_similarities = np.array(tfidf_similarities)

    absolute_error = np.absolute(similarities - tfidf_similarities)
    if square:
        absolute_error = np.square(absolute_error)
    mae = np.mean(absolute_error)
    mae_std = np.std(absolute_error)
    mae_var = np.var(absolute_error)

    return mae, mae_std, mae_var


def root_mean_squared_error(similarities, tfidf_similarities):
    """
    Root Mean Squared Error (RMSE).
    """
    mean_square_error, _, _ = mean_absolute_error(
        similarities,
        tfidf_similarities,
        square=True
    )
    root = sqrt(mean_square_error)
    return root


def apply_metric(metric_type, original_similarities, tfidf_similarities):
    metric = {
        MAE: mean_absolute_error,
        RMSE: root_mean_squared_error
    }

    return metric[metric_type](original_similarities, tfidf_similarities)


def main():
    num_songs, min_tfidf, metric_type, matching_algorithm = process_args()

    path = f"{num_songs}_songs/similarities"
    similarities = load_structure(
        structure_name=f"{path}/{matching_algorithm}",
        as_numpy=False
    )
    tfidf_similarities = load_structure(
        structure_name=f"{path}/{matching_algorithm}_min_tfidf_{min_tfidf}",
        as_numpy=False
    )

    result = apply_metric(
        metric_type,
        list(similarities.values()),
        list(tfidf_similarities.values())
    )

    evaluation_path = f"{num_songs}_songs/evaluations"
    dump_structure(
        structure=result,
        structure_name=f"{evaluation_path}/{matching_algorithm}_{metric_type}_min_tfidf_{min_tfidf}"
    )

if __name__ == "__main__":
    main()
