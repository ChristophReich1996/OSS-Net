import torch
import numpy as np


def make_pointcloud_obj(locs: torch.Tensor, actual: torch.Tensor, volume: torch.Tensor, side_len: int,
                        path: str) -> None:
    """
    Plots a given pair of point cloud predictions and labels.
    Taken from:
    :param locs: (torch.Tensor) Predicted locations
    :param actual: (torch.Tensor) Predicted label
    :param volume: (torch.Tensor) Input volume to estimate shape of obj plot
    :param side_len: (torch.Tensor) Scale parameter
    :param path: (torch.Tensor) Path to store plot
    """
    to_write = locs.cpu().numpy().astype(np.short)
    to_write_act = actual.cpu().numpy().astype(np.short)
    # Mean (shape) centering
    mean = np.array([volume.shape[2] * side_len / 2, volume.shape[3] * side_len / 2, volume.shape[4] * side_len / 2])
    to_write_act = to_write_act - mean
    to_write = to_write - mean

    with open((path + '_prediction.obj'), 'w') as f:
        for line in to_write:
            f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) +
                    " " + "0.5" + " " + "0.5" + " " + "1.0" + "\n")
        # Corners of volume
        f.write("v " + " " + "0" + " " + "0" + " " + "0" +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

        f.write("v " + " " + str(volume.shape[2] * side_len) + " " + "0" + " " + "0" +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

        f.write("v " + " " + str(volume.shape[2] * side_len) + " " + str(volume.shape[3] * side_len) + " " + "0" +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

        f.write("v " + " " + "0" + " " + str(volume.shape[3] * side_len) + " " + "0" +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

        f.write("v " + " " + "0" + " " + "0" + " " + str(volume.shape[4] * side_len) +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

        f.write("v " + " " + str(volume.shape[2] * side_len) + " " + "0" + " " + str(volume.shape[4] * side_len) +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

        f.write("v " + " " + str(volume.shape[2] * side_len) + " " + str(volume.shape[3] * side_len) + " " + str(
            volume.shape[4] * side_len) +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

        f.write("v " + " " + "0" + " " + str(volume.shape[3] * side_len) + " " + str(volume.shape[4] * side_len) +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

    with open((path + '_label.obj'), 'w') as f:
        for line in to_write_act:
            f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) +
                    " " + "0.19" + " " + "0.8" + " " + "0.19" + "\n")

        # Corners of volume
        f.write("v " + " " + "0" + " " + "0" + " " + "0" +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

        f.write("v " + " " + str(volume.shape[2] * side_len) + " " + "0" + " " + "0" +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

        f.write("v " + " " + str(volume.shape[2] * side_len) + " " + str(volume.shape[3] * side_len) + " " + "0" +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

        f.write("v " + " " + "0" + " " + str(volume.shape[3] * side_len) + " " + "0" +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

        f.write("v " + " " + "0" + " " + "0" + " " + str(volume.shape[4] * side_len) +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

        f.write("v " + " " + str(volume.shape[2] * side_len) + " " + "0" + " " + str(volume.shape[4] * side_len) +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

        f.write("v " + " " + str(volume.shape[2] * side_len) + " " + str(volume.shape[3] * side_len) + " " + str(
            volume.shape[4] * side_len) +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

        f.write("v " + " " + "0" + " " + str(volume.shape[3] * side_len) + " " + str(volume.shape[4] * side_len) +
                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
