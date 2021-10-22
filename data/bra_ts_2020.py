from typing import Tuple, Optional, List, Union, Callable
import random
import math
import os

import torch
import torch.nn.functional as F
import nibabel
from torch.utils.data import Dataset
import numpy as np

import misc
from data import augmentation


class BraTS2020SegmentationTraining(Dataset):
    """
    This class implements the BraTS 2020 dataset (training) for 3d occupancy semantic segmentation of brain MRI images.
    """

    def __init__(self, path: str,
                 normalize_coordinates: bool = False,
                 volume_shape: Tuple[int, int, int] = (240, 240, 155),
                 padding_shape: Tuple[int, int, int] = (256, 256, 160),
                 samples: Optional[int] = 2 ** 14,
                 border_balance: float = 0.5,
                 binary: bool = True,
                 number_of_classes: int = 4,
                 patches: bool = True,
                 patch_size: int = 7,
                 large_patches: bool = True,
                 augmentations: Optional[Tuple[augmentation.Augmentation]] = (
                         augmentation.RandomFlipping(dimensions=(-2, -3)),
                         augmentation.AdjustBrightness(brightness_factor=0.15),
                         augmentation.GaussianNoise(standard_deviation=0.15)),
                 augmentation_p: float = 0.5,
                 return_segmentation: bool = True,
                 segmentation_shape: Tuple[int, int, int] = (8, 8, 5),
                 downscale: int = 2,
                 return_volume_label: bool = False,
                 normalization_function: Callable = misc.normalize) -> None:
        """
        Constructor method.
        :param path: (str) Path to dataset
        :param normalize_coordinates: (bool) If true coordinates are normalizes into a range of zero to one
        :param volume_shape: (Tuple[int, int, int]) Shape of the MRI volume images
        :param padding_shape: (Tuple[int, int, int]) Shape desired after padding the MRI volume
        :param samples: (Optional[int]) Number of coordinates samples from the volume
        :param border_balance: (float) Fraction samples from inside the lesion volume
        :param binary: (bool) Binary classification label is utilized
        :param number_of_classes: (int) Number of classes for multi-class case
        :param patches: (bool) If true a small 3d patch for each sampled coordinate is returned
        :param patch_size: (int) Size of the patch (only odd numbers aloud)
        :param large_patches: (bool) If true large patches are also utilized and returned with patches
        :param augmentations: (Optional[Tuple[augmentation.Augmentation]]) Tuple of augmentations to be utilized
        :param augmentation_p: (float) Probability of applying an augmentation
        :param return_segmentation: (bool) If true the downscaled segmentation volume is returned
        :param segmentation_shape: (Tuple[int, int, int]) Shape of the segmentation label
        :param downscale: (int) Downscale factor of the MRI volume
        :param return_volume_label: (int) If true only input volume and the volume label is returned
        :param normalization_function: (Callable) Normalization function to be applied
        """
        # Save parameters
        self.normalize_coordinates = normalize_coordinates
        self.volume_shape = volume_shape
        self.padding_shape = padding_shape
        self.samples = samples
        self.border_balance = border_balance
        self.binary = binary
        self.number_of_classes = number_of_classes
        self.patches = patches
        assert (patch_size % 2) == 1, "Patch size must be an odd number."
        self.patch_size = patch_size
        self.large_patches = large_patches
        self.augmentations = augmentations
        self.augmentation_p = augmentation_p
        self.return_segmentation = return_segmentation
        self.segmentation_shape = segmentation_shape
        self.downscale = downscale
        self.return_volume_label = return_volume_label
        self.normalization_function = normalization_function
        # Get paths to data samples
        self.path_to_samples = []
        for folder in os.listdir(path=path):
            if os.path.isdir(os.path.join(path, folder)):
                self.path_to_samples.append(os.path.join(path, folder))

    def __len__(self) -> int:
        """
        Method returns the length of the dataset.
        :return: (int) Dataset length
        """
        return len(self.path_to_samples)

    def denormalize_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        This method denormalizes a given, to [0, 1] normalized tensor of coordinates.
        :param coordinates: (torch.Tensor) Normalized [0, 1] coordinates
        :return: (torch.Tensor) Denormalized coordinates
        """
        if self.normalize_coordinates:
            if coordinates.ndimension() == 2:
                return (coordinates * (
                        torch.tensor(self.padding_shape, device=coordinates.device).view(1, 3) - 1)).long()
            else:
                return (coordinates * (
                        torch.tensor(self.padding_shape, device=coordinates.device).view(1, 1, 3) - 1)).long()
        else:
            return coordinates

    @torch.no_grad()
    def __getitem__(self, item: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Method returns one sample of the dataset.
        :param item: (int) Index of the dataset sample
        :return: (Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor]]) Either 3D input volume, coordinates, labels, patches, and
        low-res. segmentation or 3D input volume and corresponding 3D label
        """
        # Load data
        data = []
        for file in sorted(os.listdir(self.path_to_samples[item])):
            if ".pt" in file:
                data.append(torch.load(os.path.join(self.path_to_samples[item], file)))
            else:
                data.append(
                    torch.from_numpy(
                        nibabel.load(os.path.join(self.path_to_samples[item], file)).get_fdata().astype(np.float32)))
        # Normalize data
        for index, normalize in zip(range(len(data)), [False, True, False, True, True, True]):
            if normalize:
                data[index] = self.normalization_function(data[index])
        # Perform augmentation
        if (self.augmentations is not None) and (self.augmentation_p > 0):
            for augmentation in self.augmentations:
                if random.random() < self.augmentation_p:
                    data = augmentation(tensors=data, apply_augmentation=[False, True, False, True, True, True])
        # Unpack data
        border, flair, label_volume, t1, t1ce, t2 = data
        # Clamp label to have four classes (0, 1, 2, 3)
        label_volume = label_volume.clamp(max=3)
        # Construct 3d MRI volume with all channels
        mri_volume = torch.stack([flair, t1, t1ce, t2], dim=0)
        # Pad MRI volume to the desired shape
        mri_volume = F.pad(mri_volume, pad=[math.ceil((self.padding_shape[2] - self.volume_shape[2]) / 2),
                                            math.trunc((self.padding_shape[2] - self.volume_shape[2]) / 2),
                                            math.ceil((self.padding_shape[1] - self.volume_shape[1]) / 2),
                                            math.trunc((self.padding_shape[1] - self.volume_shape[1]) / 2),
                                            math.ceil((self.padding_shape[0] - self.volume_shape[0]) / 2),
                                            math.trunc((self.padding_shape[0] - self.volume_shape[0]) / 2),
                                            0, 0],
                           mode="constant", value=0)
        # Pad label
        label_volume = F.pad(label_volume, pad=[math.ceil((self.padding_shape[2] - self.volume_shape[2]) / 2),
                                                math.trunc((self.padding_shape[2] - self.volume_shape[2]) / 2),
                                                math.ceil((self.padding_shape[1] - self.volume_shape[1]) / 2),
                                                math.trunc((self.padding_shape[1] - self.volume_shape[1]) / 2),
                                                math.ceil((self.padding_shape[0] - self.volume_shape[0]) / 2),
                                                math.trunc((self.padding_shape[0] - self.volume_shape[0]) / 2)],
                             mode="constant", value=0)
        # Return only input and label volumes
        if self.return_volume_label:
            # Case if binary classification is utilized
            if self.binary:
                label_volume = (label_volume > 0).unsqueeze(dim=0).float()
            # Multi-class case
            else:
                label_volume = F.one_hot(label_volume.long(), num_classes=self.number_of_classes).float()
                label_volume = label_volume.permute(-1, 0, 1, 2)
            return mri_volume, label_volume
        # Pad border
        border = F.pad(border, pad=[math.ceil((self.padding_shape[2] - self.volume_shape[2]) / 2),
                                    math.trunc((self.padding_shape[2] - self.volume_shape[2]) / 2),
                                    math.ceil((self.padding_shape[1] - self.volume_shape[1]) / 2),
                                    math.trunc((self.padding_shape[1] - self.volume_shape[1]) / 2),
                                    math.ceil((self.padding_shape[0] - self.volume_shape[0]) / 2),
                                    math.trunc((self.padding_shape[0] - self.volume_shape[0]) / 2)],
                       mode="constant", value=0)
        # Get coordinates of lesions
        coordinates_lesion = torch.from_numpy(np.argwhere(border.numpy() > 0))
        # Sample coordinates
        if coordinates_lesion.shape[0] > int(self.samples * self.border_balance):
            coordinates_lesion = coordinates_lesion[
                torch.randperm(coordinates_lesion.shape[0])[:int(self.samples * self.border_balance)]]
            coordinates_no_lesion = (torch.rand(math.ceil(self.samples * (1. - self.border_balance)), 3)
                                     * (torch.tensor(label_volume.shape).view(1, -1) - 1)).long()
        else:
            samples_lesion = coordinates_lesion.shape[0]
            samples_no_lesion = self.samples - samples_lesion
            coordinates_lesion = coordinates_lesion[
                torch.randperm(coordinates_lesion.shape[0])[:samples_lesion]]
            coordinates_no_lesion = (torch.rand(samples_no_lesion, 3)
                                     * (torch.tensor(label_volume.shape).view(1, -1) - 1)).long()
        # Concat coordinates
        coordinates = torch.cat([coordinates_lesion, coordinates_no_lesion], dim=0)
        # Shuffle coordinates
        coordinates = coordinates[torch.randperm(coordinates.shape[0])]
        # Get label for each coordinate
        labels = label_volume[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]
        # Case if binary classification is utilized
        if self.binary:
            labels = (labels > 0).unsqueeze(dim=-1).float()
        # Multi-class case
        else:
            labels = F.one_hot(labels.long(), num_classes=self.number_of_classes).float()
        # Case if patches are utilized
        if self.patches:
            # Pad volume to extract patches
            mri_volume_padded = F.pad(mri_volume, pad=[self.patch_size // 2, self.patch_size // 2,
                                                       self.patch_size // 2, self.patch_size // 2,
                                                       self.patch_size // 2, self.patch_size // 2,
                                                       0, 0], mode="constant", value=0)
            # Get all possible patches
            patches = mri_volume_padded \
                .unfold(dimension=1, size=self.patch_size, step=1) \
                .unfold(dimension=2, size=self.patch_size, step=1) \
                .unfold(dimension=3, size=self.patch_size, step=1)
            # Get patches corresponding to each coordinate, reshape and flatten
            patches = patches[:, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] \
                .transpose(0, 1).float()
            if self.large_patches:
                # Pad volume to extract patches
                mri_volume_padded = F.pad(mri_volume, pad=[self.patch_size, self.patch_size - 1,
                                                           self.patch_size, self.patch_size - 1,
                                                           self.patch_size, self.patch_size - 1,
                                                           0, 0], mode="constant", value=0)
                # Get all possible patches
                patches_large = mri_volume_padded \
                    .unfold(dimension=1, size=self.patch_size * 2, step=1) \
                    .unfold(dimension=2, size=self.patch_size * 2, step=1) \
                    .unfold(dimension=3, size=self.patch_size * 2, step=1)
                # Get patches corresponding to each coordinate, reshape and flatten
                patches_large = patches_large[:, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] \
                    .transpose(0, 1).float()
                # Perform max pooling
                patches_large = F.avg_pool3d(patches_large, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                # Concat large and local patches
                patches = torch.cat([patches, patches_large], dim=1)
        else:
            patches = None
        # Normalize coordinates of utilized
        if self.normalize_coordinates:
            coordinates = coordinates.float() / (torch.tensor(self.padding_shape).view(1, -1) - 1)
        # Downscale segmentation if utilized
        if self.return_segmentation:
            segmentation = F.adaptive_max_pool3d((label_volume[None, None] > 0).float(),
                                                 output_size=self.segmentation_shape)[0]
        else:
            segmentation = None
        # Downscale MRI volume if utilized
        if self.downscale > 1:
            mri_volume = F.interpolate(mri_volume[None].float(),
                                       size=(self.padding_shape[-3] // self.downscale,
                                             self.padding_shape[-2] // self.downscale,
                                             self.padding_shape[-1] // self.downscale),
                                       mode="trilinear", align_corners=False)[0]
        return mri_volume.float(), coordinates.float(), labels, patches, segmentation


class BraTS2020SegmentationTest(BraTS2020SegmentationTraining):
    """
    This class implements the BraTS 2020 dataset (test) for 3d occupancy semantic segmentation of brain MRI images.
    """

    def __init__(self, path: str,
                 normalize_coordinates: bool = False,
                 volume_shape: Tuple[int, int, int] = (240, 240, 155),
                 padding_shape: Tuple[int, int, int] = (256, 256, 160),
                 samples: int = 2 ** 17,
                 binary: bool = True,
                 number_of_classes: int = 4,
                 patches: bool = True,
                 patch_size: int = 7,
                 downscale: int = 2,
                 return_volume_label: bool = False,
                 large_patches: bool = True,
                 normalization_function: Callable = misc.normalize) -> None:
        """
        Constructor method.
        :param path: (str) Path to dataset
        :param normalize_coordinates: (bool) If true coordinates are normalizes into a range of zero to one
        :param volume_shape: (Tuple[int, int, int]) Shape of the MRI volume images
        :param padding_shape: (Tuple[int, int, int]) Shape desired after padding the MRI volume
        :param samples: (int) Number of coordinates samples from the volume
        :param binary: (bool) Binary classification label is utilized
        :param number_of_classes: (int) Number of classes for multi-class case
        :param patches: (bool) If true a small 3d patch for each sampled coordinate is returned
        :param patch_size: (int) Size of the patch (only odd numbers aloud)
        :param downscale: (int) Downscale factor of the MRI volume
        :param return_volume_label: (int) If true only input volume and the volume label is returned
        :param large_patches: (bool) If true large patches are also utilized and returned with patches
        :param normalization_function: (Callable) Normalization function to be applied
        """
        super().__init__(path=path, normalize_coordinates=normalize_coordinates, volume_shape=volume_shape,
                         padding_shape=padding_shape, samples=samples, border_balance=0.0, binary=binary,
                         number_of_classes=number_of_classes, patches=patches, patch_size=patch_size,
                         augmentations=None, augmentation_p=0.0, return_segmentation=False, downscale=downscale,
                         return_volume_label=return_volume_label, large_patches=large_patches,
                         normalization_function=normalization_function)


class BraTS2020SegmentationInference(BraTS2020SegmentationTraining):
    """
    This class implements the BraTS 2020 dataset (test) for 3d occupancy semantic segmentation of brain MRI images.
    """

    def __init__(self, path: str,
                 normalize_coordinates: bool = False,
                 volume_shape: Tuple[int, int, int] = (240, 240, 155),
                 padding_shape: Tuple[int, int, int] = (256, 256, 160),
                 binary: bool = True,
                 number_of_classes: int = 4,
                 large_patches: bool = True,
                 patches: bool = True,
                 patch_size: int = 7,
                 downscale: int = 2,
                 normalization_function: Callable = misc.normalize) -> None:
        """
        Constructor method.
        :param path: (str) Path to dataset
        :param normalize_coordinates: (bool) If true coordinates are normalizes into a range of zero to one
        :param volume_shape: (Tuple[int, int, int]) Shape of the MRI volume images
        :param padding_shape: (Tuple[int, int, int]) Shape desired after padding the MRI volume
        :param samples: (int) Number of coordinates samples from the volume
        :param binary: (bool) Binary classification label is utilized
        :param number_of_classes: (int) Number of classes for multi-class case
        :param large_patches: (bool) If true large patches are also utilized and returned with patches
        :param patches: (bool) If true a small 3d patch for each sampled coordinate is returned
        :param patch_size: (int) Size of the patch (only odd numbers aloud)
        :param downscale: (int) Downscale factor of the MRI volume
        :param normalization_function: (Callable) Normalization function to be applied
        """
        super().__init__(path=path, normalize_coordinates=normalize_coordinates, volume_shape=volume_shape,
                         padding_shape=padding_shape, samples=None, border_balance=0.0, binary=binary,
                         number_of_classes=number_of_classes, patches=patches, patch_size=patch_size,
                         augmentations=None, augmentation_p=0.0, return_segmentation=False, downscale=downscale,
                         large_patches=large_patches, normalization_function=normalization_function)

    @torch.no_grad()
    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Method returns one sample of the dataset.
        :param item: (int) Index of the dataset sample
        :return: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) 3d MRI volume, coordinates, patches and
        large patches.
        """
        # Load data
        data = []
        for file in sorted(os.listdir(self.path_to_samples[item])):
            if ".pt" in file:
                data.append(torch.load(os.path.join(self.path_to_samples[item], file)).float())
            else:
                data.append(
                    torch.from_numpy(
                        nibabel.load(os.path.join(self.path_to_samples[item], file)).get_fdata().astype(np.float32)))
        # Unpack data
        border, flair, label_volume, t1, t1ce, t2 = data
        # Normalize input data
        flair = self.normalization_function(flair)
        t1 = self.normalization_function(t1)
        t1ce = self.normalization_function(t1ce)
        t2 = self.normalization_function(t2)
        # Construct 3d MRI volume with all channels
        mri_volume = torch.stack([flair, t1, t1ce, t2], dim=0)
        # Pad MRI volume to the desired shape
        mri_volume = F.pad(mri_volume, pad=[math.ceil((self.padding_shape[2] - self.volume_shape[2]) / 2),
                                            math.trunc((self.padding_shape[2] - self.volume_shape[2]) / 2),
                                            math.ceil((self.padding_shape[1] - self.volume_shape[1]) / 2),
                                            math.trunc((self.padding_shape[1] - self.volume_shape[1]) / 2),
                                            math.ceil((self.padding_shape[0] - self.volume_shape[0]) / 2),
                                            math.trunc((self.padding_shape[0] - self.volume_shape[0]) / 2),
                                            0, 0],
                           mode="constant", value=0)
        # Pad label
        label_volume = F.pad(label_volume, pad=[math.ceil((self.padding_shape[2] - self.volume_shape[2]) / 2),
                                                math.trunc((self.padding_shape[2] - self.volume_shape[2]) / 2),
                                                math.ceil((self.padding_shape[1] - self.volume_shape[1]) / 2),
                                                math.trunc((self.padding_shape[1] - self.volume_shape[1]) / 2),
                                                math.ceil((self.padding_shape[0] - self.volume_shape[0]) / 2),
                                                math.trunc((self.padding_shape[0] - self.volume_shape[0]) / 2)],
                             mode="constant", value=0)
        # Make patches
        if self.patches:
            # Pad volume to extract patches
            mri_volume_padded = F.pad(mri_volume, pad=[self.patch_size // 2, self.patch_size // 2,
                                                       self.patch_size // 2, self.patch_size // 2,
                                                       self.patch_size // 2, self.patch_size // 2,
                                                       0, 0], mode="constant", value=0)
            # Get all possible patches
            patches = mri_volume_padded \
                .unfold(dimension=1, size=self.patch_size, step=1) \
                .unfold(dimension=2, size=self.patch_size, step=1) \
                .unfold(dimension=3, size=self.patch_size, step=1)
        else:
            patches = None
        # Make large patches
        if self.large_patches:
            # Pad volume to extract patches
            mri_volume_padded = F.pad(mri_volume, pad=[self.patch_size, self.patch_size - 1,
                                                       self.patch_size, self.patch_size - 1,
                                                       self.patch_size, self.patch_size - 1,
                                                       0, 0], mode="constant", value=0)
            # Get all possible patches
            large_patches = mri_volume_padded \
                .unfold(dimension=1, size=self.patch_size * 2, step=1) \
                .unfold(dimension=2, size=self.patch_size * 2, step=1) \
                .unfold(dimension=3, size=self.patch_size * 2, step=1)
        else:
            large_patches = None
        # Case if binary classification is utilized
        if self.binary:
            label = (label_volume > 0).float()
        # Multi-class case
        else:
            label = F.one_hot(label_volume.long(), num_classes=self.number_of_classes).float()
        # Downscale MRI volume if utilized
        if self.downscale > 1:
            mri_volume = F.interpolate(mri_volume[None].float(),
                                       size=(self.padding_shape[-3] // self.downscale,
                                             self.padding_shape[-2] // self.downscale,
                                             self.padding_shape[-1] // self.downscale),
                                       mode="trilinear", align_corners=False)[0]
        # Add batch dim if needed
        mri_volume = mri_volume.unsqueeze(dim=0) if mri_volume is not None else mri_volume
        patches = patches.unsqueeze(dim=0) if patches is not None else patches
        label = label.unsqueeze(dim=0) if label is not None else label
        large_patches = large_patches.unsqueeze(dim=0) if large_patches is not None else large_patches
        return mri_volume.float(), patches, label, large_patches
