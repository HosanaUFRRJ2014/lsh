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

    parser.add_argument(
        "--metric_type",
        "-metric",
        type=str,
        help=f"Metric to be applied. Options: {METRIC_TYPES}"
    )

    parser.add_argument(
        "--min_tfidf",
        "-min",
        type=float,
        help=f"Gets the previous calculated jaccard similarities for this minimum TF-IDF."
    )

    args = parser.parse_args()

    min_tfidf = args.min_tfidf
    metric_type = args.metric_type
    return min_tfidf, metric_type


def mean_absolute_error(jaccard_similarities, tfidf_jaccard_similarities, square=False):
    """
    Mean Absolute Error (MAE).
    Calculates MAE between jaccard similarities and jaccard similarities
    after aplication of TF-IDF.
    """
    if not isinstance(jaccard_similarities, np.ndarray):
        jaccard_similarities = np.array(jaccard_similarities)
    
    if not isinstance(tfidf_jaccard_similarities, np.ndarray):
        tfidf_jaccard_similarities = np.array(tfidf_jaccard_similarities)

    absolute_error = np.absolute(jaccard_similarities - tfidf_jaccard_similarities)
    if square:
        absolute_error = np.square(absolute_error)
    mae = np.mean(absolute_error)

    return mae


def root_mean_squared_error(jaccard_similarities, tfidf_jaccard_similarities):
    """
    Root Mean Squared Error (RMSE).
    """
    mean_square_error = mean_absolute_error(
        jaccard_similarities,
        tfidf_jaccard_similarities,
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
    min_tfidf, metric_type = process_args()

    jaccard_similarities = load_structure(
        structure_name="jaccard_similarities",
        as_numpy=False
    )
    tfidf_jaccard_similarities = load_structure(
        structure_name=f"jaccard_similarities_min_tfidf_{min_tfidf}",
        as_numpy=False
    )

    result = apply_metric(
        metric_type,
        np.array(list(jaccard_similarities.values())),
        np.array(list(tfidf_jaccard_similarities.values()))
    )

    dump_structure(
        structure=result,
        structure_name=f"{metric_type}_min_tfidf_{min_tfidf}",
        extension="txt"
    )

if __name__ == "__main__":
    main()
