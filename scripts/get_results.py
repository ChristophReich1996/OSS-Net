# Script to print results of multiple experiments
from typing import List
import os

import torch
import numpy as np


def get_paths_to_data(path: str, model: str, metric: str, dataset: str) -> List[str]:
    """
    Function gets paths to results.
    :param path: (str) Path to results
    :param model: (str) Name of model
    :param metric: (str) Name of metric
    :return: (List[str]) List of all training runs metric
    """
    # Init list to store paths to metric
    metric_paths = []
    # Iterate over all training runs
    for folder in os.listdir(path=path):
        # Check if model us utilized
        if model in folder and dataset in folder:
            # Iterate over metrics
            print(folder)
            for file in os.listdir(path=os.path.join(path, folder, "metrics")):
                if metric in file:
                    metric_paths.append(os.path.join(path, folder, "metrics", file))
    return metric_paths


def load_and_average(path_to_metrics: List[str]) -> None:
    """
    Method loads and averages metric over multiple runs
    :param path_to_metrics: (List[str]) Paths to metrics of different training runs
    """
    # Init list to store metrics
    metric = []
    # Load metrics
    for path in path_to_metrics:
        metric.append(torch.load(path).max().item())
    if len(metric) > 0:
        # Print metric
        print(np.nanargmax(metric), np.mean(metric), np.std(metric))


def main() -> None:
    # Init path to date
    path = "F:/OSS-Net_results/weighted_sampling"
    # Init models
    models = ["_full2_", "_full_", "_B_", "_A_"]
    # Init metrics
    metrics = ["IoU", "Dice"]
    # Init dataset name
    dataset = "LITS"
    # Iterate over all models
    for model in models:
        print(model)
        # Iterate over all metrics
        for metric in metrics:
            print(metric)
            load_and_average(get_paths_to_data(path=path, model=model, metric=metric, dataset=dataset))


if __name__ == '__main__':
    main()
