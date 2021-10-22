# Import BraTS 2020 datasets
from .bra_ts_2020 import BraTS2020SegmentationTraining, BraTS2020SegmentationTest, BraTS2020SegmentationInference
# Import LITS datasets
from .lits import LITSSegmentationTraining, LITSSegmentationTest, LITSSegmentationInference
# Imports for functions
from typing import List, Tuple, Optional
import torch


def collate_function(
        data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Collate function for dataloader of BraTS2020Segmentation datasets.
    :param data: (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]])
    List of data samples
    :return: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]])
    Batched data
    """
    # Constructed batch of MRI volumes
    mri_volumes = torch.stack([data[index][0] for index in range(len(data))], dim=0)
    # Constructed batch of coordinates by concatenation
    coordinates = torch.stack([data[index][1] for index in range(len(data))], dim=0)
    # Constructed batch of labels by concatenation
    labels = torch.stack([data[index][2] for index in range(len(data))], dim=0)
    # Case if patches are given
    if data[0][3] is not None:
        patches = torch.stack([data[index][3] for index in range(len(data))], dim=0)
    else:
        patches = None
    # Case if segmentation's are given
    if data[0][4] is not None:
        segmentations = torch.stack([data[index][4] for index in range(len(data))], dim=0)
    else:
        segmentations = None
    return mri_volumes, coordinates, labels, patches, segmentations


def collate_function_height(data: List[Tuple[torch.Tensor, torch.Tensor]],
                            slices: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for U-Net 2D (height slicing).
    :param data: (List[Tuple[torch.Tensor, torch.Tensor]]) Batch as list of volume input and label
    :param slices: (int) Number of slices taken from each sample, if none whole input is returned
    :return: (Tuple[torch.Tensor, torch.Tensor]) Batched output [batch size, channels, height, width]
    """
    # Init lists to store inputs and labels
    inputs: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    # Iterate over batches
    for index, sample in enumerate(data):
        if slices is not None:
            indexes: torch.Tensor = torch.randint(low=0, high=sample[0].shape[1], size=(slices,))
            inputs.append(sample[0][:, indexes])
            labels.append(sample[1][:, indexes])
        else:
            inputs.append(sample[0])
            labels.append(sample[1])
    # Stack output
    inputs: torch.Tensor = torch.stack(inputs, dim=0)
    labels: torch.Tensor = torch.stack(labels, dim=0)
    # Reshape from [batch size, channels, height, width, slices] to [batch size * slices, channels, height, width]
    inputs: torch.Tensor = inputs.permute(0, 2, 1, 3, 4).contiguous().flatten(start_dim=0, end_dim=1)
    labels: torch.Tensor = labels.permute(0, 2, 1, 3, 4).contiguous().flatten(start_dim=0, end_dim=1)
    return inputs, labels


def collate_function_width(data: List[Tuple[torch.Tensor, torch.Tensor]],
                           slices: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for U-Net 2D (width slicing).
    :param data: (List[Tuple[torch.Tensor, torch.Tensor]]) Batch as list of volume input and label
    :param slices: (int) Number of slices taken from each sample, if none whole input is returned
    :return: (Tuple[torch.Tensor, torch.Tensor]) Batched output [batch size, channels, height, width]
    """
    # Init lists to store inputs and labels
    inputs: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    # Iterate over batches
    for index, sample in enumerate(data):
        if slices is not None:
            indexes: torch.Tensor = torch.randint(low=0, high=sample[0].shape[-1], size=(slices,))
            inputs.append(sample[0][:, :, indexes])
            labels.append(sample[1][:, :, indexes])
        else:
            inputs.append(sample[0])
            labels.append(sample[1])
    # Stack output
    inputs: torch.Tensor = torch.stack(inputs, dim=0)
    labels: torch.Tensor = torch.stack(labels, dim=0)
    # Reshape from [batch size, channels, height, width, slices] to [batch size * slices, channels, height, width]
    inputs: torch.Tensor = inputs.permute(0, 3, 1, 2, 4).contiguous().flatten(start_dim=0, end_dim=1)
    labels: torch.Tensor = labels.permute(0, 3, 1, 2, 4).contiguous().flatten(start_dim=0, end_dim=1)
    return inputs, labels


def collate_function_depth(data: List[Tuple[torch.Tensor, torch.Tensor]],
                           slices: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for U-Net 2D (depth slicing).
    :param data: (List[Tuple[torch.Tensor, torch.Tensor]]) Batch as list of volume input and label
    :param slices: (int) Number of slices taken from each sample, if none whole input is returned
    :return: (Tuple[torch.Tensor, torch.Tensor]) Batched output [batch size, channels, height, width]
    """
    # Init lists to store inputs and labels
    inputs: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    # Iterate over batches
    for index, sample in enumerate(data):
        if slices is not None:
            indexes: torch.Tensor = torch.randint(low=0, high=sample[0].shape[-1], size=(slices,))
            inputs.append(sample[0][..., indexes])
            labels.append(sample[1][..., indexes])
        else:
            inputs.append(sample[0])
            labels.append(sample[1])
    # Stack output
    inputs: torch.Tensor = torch.stack(inputs, dim=0)
    labels: torch.Tensor = torch.stack(labels, dim=0)
    # Reshape from [batch size, channels, height, width, slices] to [batch size * slices, channels, height, width]
    inputs: torch.Tensor = inputs.permute(0, 4, 1, 2, 3).contiguous().flatten(start_dim=0, end_dim=1)
    labels: torch.Tensor = labels.permute(0, 4, 1, 2, 3).contiguous().flatten(start_dim=0, end_dim=1)
    return inputs, labels
