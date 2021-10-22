from typing import Tuple, Union, List, Any

import torch
import numpy as np


class Augmentation(object):
    """
    Super class for all augmentations.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """
        Call method is used to apply the augmentation
        :param args: Will be ignored
        :param kwargs: Will be ignored
        """
        raise NotImplementedError()


class RandomFlipping(Augmentation):
    """
    Random flipping augmentation over given dimensions
    """

    def __init__(self, dimensions: Union[Tuple[int, ...], int]) -> None:
        """
        Constructor method
        :param dimensions: (Union[Tuple[int, ...], int]) Dimensions to be utilized
        """
        # Call super constructor
        super(RandomFlipping, self).__init__()
        # Save parameter
        self.dimensions = dimensions if isinstance(dimensions, tuple) else (dimensions,)

    def __call__(self, tensors: List[torch.Tensor], *args, **kwargs) -> List[torch.Tensor]:
        """
        Call method is used to apply the augmentation
        :param tensors: (Tuple[torch.Tensor, ...]) Tensors to be augmented by random flipping
        :param args: Will be ignored
        :param kwargs: Will be ignored
        """
        # Choose dimension randomly
        dimension = tuple(
            np.random.choice(self.dimensions, size=np.random.randint(1, len(self.dimensions) + 1, 1), replace=False))
        # Augment all tensors in given list
        for index in range(len(tensors)):
            # Perform flipping
            tensors[index] = flip(tensor=tensors[index], dims=dimension)
        return tensors


class GaussianNoise(Augmentation):
    """
    This class implements gaussian noise injection augmentation.
    """

    def __init__(self, standard_deviation: float = 0.1) -> None:
        """
        Constructor method
        :param standard_deviation: (float) Standard deviation of gaussian noise to be added
        """
        # Call super constructor
        super(GaussianNoise, self).__init__()
        # Save parameter
        self.standard_deviation = standard_deviation

    def __call__(self, tensors: List[torch.Tensor], apply_augmentation: List[bool], *args: Any,
                 **kwargs: Any) -> List[torch.Tensor]:
        """
        Call method is used to apply the augmentation
        :param tensors: (List[torch.Tensor]) Tensors to be augmented by random flipping
        :param apply: (List[bool]) List of booleans if true augmentation is to the corresponding tensor
        :param args: Will be ignored
        :param kwargs: Will be ignored
        """
        # Augment all tensors in given list
        for index, (tensor, apply) in enumerate(zip(tensors, apply_augmentation)):
            if apply:
                # Add gaussian noise
                tensors[index] = tensor + torch.randn_like(tensor) * self.standard_deviation
        return tensors


class AdjustBrightness(Augmentation):
    """
    This class implements random brightness adjustment augmentation.
    """

    def __init__(self, brightness_factor: float = 0.15) -> None:
        """
        Constructor method
        :param brightness_factor: (float) Max brightness change relative to the max value of the tensor
        """
        # Call super constructor
        super(AdjustBrightness, self).__init__()
        # Save parameter
        self.brightness_factor = brightness_factor

    def __call__(self, tensors: List[torch.Tensor], apply_augmentation: List[bool], *args,
                 **kwargs) -> List[torch.Tensor]:
        """
        Call method is used to apply the augmentation
        :param tensors: (Tuple[torch.Tensor, ...]) Tensors to be augmented by random flipping
        :param apply: (Tuple[bool]) List of booleans if true augmentation is to the corresponding tensor
        :param args: Will be ignored
        :param kwargs: Will be ignored
        """
        # Choose brightness adjustment value
        brightness_factor = (2. * torch.rand(len(tensors)) - 1.) * self.brightness_factor
        # Augment all tensors in given list
        for index, (tensor, apply) in enumerate(zip(tensors, apply_augmentation)):
            if apply:
                # Perform brightness adjustment
                tensors[index] = (tensor + tensor.max() * brightness_factor[index])
        return tensors


def flip(tensor: torch.Tensor, dims: Tuple[int]) -> torch.Tensor:
    """
    Function to flip a tensor at given dimensions with advanced indexing.
    Much faster than standard flip function
    :param tensor: Tensor to be flipped
    :param dim: (Tuple[int]) Dimensions to be flipped
    :return: Flipped tensor
    """
    for dim in dims:
        # Init index tensor
        reverse_index = torch.arange(tensor.shape[dim] - 1, -1, -1)
        # Perform flipping
        if dim == -1:
            tensor = tensor[..., reverse_index]
        elif dim == -2:
            tensor = tensor[..., reverse_index, :]
        elif dim == -3:
            tensor = tensor[..., reverse_index, :, :]
        else:
            raise ValueError("Illegal dim to be flip. Dim: {}".format(dim))
    return tensor
