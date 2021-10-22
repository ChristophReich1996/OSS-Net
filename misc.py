from typing import Any, Dict, Union, Iterable, Optional, Tuple
import os
import json
from datetime import datetime

import torch


class Logger(object):
    """
    Class to log different metrics.
    """

    def __init__(self,
                 experiment_path: str =
                 os.path.join(os.getcwd(), "experiments", datetime.now().strftime("%d_%m_%Y__%H_%M_%S")),
                 experiment_path_extension: str = "",
                 path_metrics: str = "metrics",
                 path_hyperparameters: str = "hyperparameters",
                 path_plots: str = "plots",
                 path_models: str = "models") -> None:
        """
        Constructor method
        :param path_metrics: (str) Path to folder in which all metrics are stored
        :param experiment_path_extension: (str) Extension to experiment folder
        :param path_hyperparameters: (str)  Path to folder in which all hyperparameters are stored
        :param path_plots: (str)  Path to folder in which all plots are stored
        :param path_models: (str)  Path to folder in which all models are stored
        """
        experiment_path = experiment_path + experiment_path_extension
        # Save parameters
        self.path_metrics = os.path.join(experiment_path, path_metrics)
        self.path_hyperparameters = os.path.join(experiment_path, path_hyperparameters)
        self.path_plots = os.path.join(experiment_path, path_plots)
        self.path_models = os.path.join(experiment_path, path_models)
        # Init folders
        os.makedirs(self.path_metrics, exist_ok=True)
        os.makedirs(self.path_hyperparameters, exist_ok=True)
        os.makedirs(self.path_plots, exist_ok=True)
        os.makedirs(self.path_models, exist_ok=True)
        # Init dicts to store the metrics and hyperparameters
        self.metrics = dict()
        self.temp_metrics = dict()
        self.hyperparameters = dict()

    def log_metric(self, metric_name: str, value: Any) -> None:
        """
        Method writes a given metric value into a dict including list for every metric.
        :param metric_name: (str) Name of the metric
        :param value: (float) Value of the metric
        """
        if metric_name in self.metrics:
            self.metrics[metric_name].append(float(value))
        else:
            self.metrics[metric_name] = [float(value)]

    def log_temp_metric(self, metric_name: str, value: Any) -> None:
        """
        Method writes a given metric value into a dict including temporal metrics.
        :param metric_name: (str) Name of the metric
        :param value: (float) Value of the metric
        """
        if metric_name in self.temp_metrics:
            self.temp_metrics[metric_name].append(float(value))
        else:
            self.temp_metrics[metric_name] = [float(value)]

    def save_temp_metric(self, metric_name: Union[Iterable[str], str]) -> Dict[str, float]:
        """
        Method writes temporal metrics into the metrics dict by averaging.
        :param metric_name: (Union[Iterable[str], str]) One temporal metric name ore a list of names
        """
        averaged_temp_dict = dict()
        # Case if only one metric is given
        if isinstance(metric_name, str):
            # Calc average
            value = float(torch.tensor(self.temp_metrics[metric_name]).mean())
            # Save metric in log dict
            self.log_metric(metric_name=metric_name, value=value)
            # Put metric also in dict to be returned
            averaged_temp_dict[metric_name] = value
        # Case if multiple metrics are given
        else:
            for name in metric_name:
                # Calc average
                value = float(torch.tensor(self.temp_metrics[name]).mean())
                # Save metric in log dict
                self.log_metric(metric_name=name, value=value)
                # Put metric also in dict to be returned
                averaged_temp_dict[name] = value
        # Reset temp metrics
        self.temp_metrics = dict()
        # Save logs
        self.save()
        return averaged_temp_dict

    def log_hyperparameter(self, hyperparameter_name: Optional[str] = None, value: Optional[Any] = None,
                           hyperparameter_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Method writes a given hyperparameter into a dict including all other hyperparameters.
        :param hyperparameter_name: (Optional[str]) Name of the hyperparameter
        :param value: (Optional[Any]) Value of the hyperparameter, must by convertible to str
        :param hyperparameter_dict: (Optional[Dict[str, Any]]) Dict of multiple hyperparameter to be saved
        """
        # Case if name and value are given
        if (hyperparameter_name is not None) and (value is not None):
            if hyperparameter_name in self.hyperparameters:
                self.hyperparameters[hyperparameter_name].append(str(value))
            else:
                self.hyperparameters[hyperparameter_name] = [str(value)]
        # Case if dict of hyperparameters is given
        if hyperparameter_dict is not None:
            # Iterate over given dict, cast data and store in internal hyperparameters dict
            for key in hyperparameter_dict.keys():
                if key in self.hyperparameters.keys():
                    self.hyperparameters[key].append(str(hyperparameter_dict[key]))
                else:
                    self.hyperparameters[key] = [str(hyperparameter_dict[key])]

    def save_occupancy_grid(self, occupancy_grid: torch.Tensor, name: str) -> None:
        """
        Method to save a given occupancy grid tensor.
        :param occupancy_grid: (torch.Tensor) Occupancy grid to be saved
        :param name: (str) Name of the file to be stored
        """
        torch.save(occupancy_grid.cpu(), os.path.join(self.path_plots, name))

    def save_mesh(self, vertices: torch.Tensor, triangles: torch.Tensor, name: Tuple[str, str]) -> None:
        """
        Method to save a given mesh of vertices and triangles
        :param vertices: (torch.Tensor) Tensor of vertices
        :param triangles: (torch.Tensor) Tensor fo triangles
        :param name: (Tuple[str]) Names of the file to be stored
        """
        torch.save(vertices.cpu(), os.path.join(self.path_plots, name[0]))
        torch.save(triangles.cpu(), os.path.join(self.path_plots, name[1]))

    def log_model(self, file_name: str, model: torch.nn.Module) -> None:
        """
        This method saves the state dict of given nn.Module.
        :param name: (str) File name with file format
        :param model: (torch.nn.Module) Module to be saved
        """
        torch.save(model.state_dict(), os.path.join(self.path_models, file_name))

    def save(self) -> None:
        """
        Method saves all current logs (metrics and hyperparameters). Plots are saved directly.
        """
        # Save dict of hyperparameter as json file
        with open(os.path.join(self.path_hyperparameters, 'hyperparameter.txt'), 'w') as json_file:
            json.dump(self.hyperparameters, json_file)
        # Iterate items in metrics dict
        for metric_name, values in self.metrics.items():
            # Convert list of values to torch tensor to use build in save method from torch
            values = torch.tensor(values)
            # Save values
            torch.save(values, os.path.join(self.path_metrics, '{}.pt'.format(metric_name)))


def normalize_0_1(input: torch.tensor) -> torch.tensor:
    """
    Normalize a given tensor to a range of [0, 1].
    :param input: (Torch tensor) Input tensor
    :return: (Torch tensor) Normalized output tensor
    """
    return (input - input.min()) / (input.max() - input.min())


def normalize_0_1_slice(input: torch.tensor) -> torch.tensor:
    """
    Normalize a given tensor slice_wise to a range of [0, 1].
    :param input: (Torch tensor) Input tensor
    :return: (Torch tensor) Normalized output tensor
    """
    original_shape = input.shape
    input = input.contiguous().view(-1, input.shape[-1])
    input = (input - input.min(dim=0, keepdim=True)[0]) / (
            input.max(dim=0, keepdim=True)[0] - input.min(dim=0, keepdim=True)[0] + 1e-05)
    return input.reshape(original_shape)


def normalize(input: torch.tensor) -> torch.tensor:
    """
    Normalize a given tensor to a mean of zero and a variance of one.
    :param input: (Torch tensor) Input tensor
    :return: (Torch tensor) Normalized output tensor
    """
    return (input - input.mean()) / input.std()


def normalize_slices(input: torch.tensor) -> torch.tensor:
    """
    Normalize a given tensor slice-wise to a mean of zero and a variance of one.
    :param input: (Torch tensor) Input tensor
    :return: (Torch tensor) Normalized output tensor
    """
    return (input - input.mean(dim=(0, 1), keepdim=True)) / input.std(dim=(0, 1), keepdim=True).clip(min=1e-05)
