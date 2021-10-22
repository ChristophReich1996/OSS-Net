import torch
import torch.nn as nn


class IoU(nn.Module):
    """
    This class implements the intersection over union for occupancy predictions.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold for segmentation prediction
        """
        # Call super constructor
        super(IoU, self).__init__()
        # Save parameter
        self.threshold = threshold

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return self.__class__.__name__

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param prediction: (torch.Tensor) One hot prediction (no offset) of the shape [batch size, coordinates, classes]
        :param label: (torch.Tensor) Labels encoded as one hot of the shape [batch size, coordinates, classes]
        :return: (torch.Tensor) Scalar dice loss
        """
        # Flatten prediction and label
        prediction = prediction.reshape(prediction.shape[0], -1)
        label = label.reshape(label.shape[0], -1)
        # Apply threshold
        prediction = (prediction > self.threshold).float()
        # Calc intersection
        intersection = ((prediction + label) == 2.0).sum(dim=-1)
        # Calc union
        union = ((prediction + label) >= 1.0).sum(dim=-1)
        # Calc IoU
        iou = (intersection / (union + 1e-10)).mean()
        return iou


class DiceScore(nn.Module):
    """
    This class implements the dice score for occupancy predictions.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold for segmentation prediction
        """
        # Call super constructor
        super(DiceScore, self).__init__()
        # Save parameter
        self.threshold = threshold

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return self.__class__.__name__

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param prediction: (torch.Tensor) One hot prediction (no offset) of the shape [batch size, coordinates, classes]
        :param label: (torch.Tensor) Labels encoded as one hot of the shape [batch size, coordinates, classes]
        :return: (torch.Tensor) Scalar dice loss
        """
        # Flatten prediction and label
        prediction = prediction.reshape(prediction.shape[0], -1)
        label = label.reshape(label.shape[0], -1)
        # Apply threshold
        prediction = (prediction > self.threshold).float()
        # Calc intersection
        intersection = ((prediction + label) == 2.0).sum(dim=-1)
        # Calc dice score for each batch instance and average over batches
        dice_score = ((2. * intersection) / (prediction.sum(dim=-1) + label.sum(dim=-1) + 1e-10)).mean()
        return dice_score
