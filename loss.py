import torch
import torch.nn as nn


class BinaryCrossEntropyLoss(nn.Module):
    """
    This class implements the binary cross entropy loss.
    """

    def __init__(self, bootstrap: bool = True, k_top: float = 0.8,
                 label_smoothing: float = 0., w_0: float = 1, w_1: float = 1) -> None:
        """
        Constructor method
        :param bootstrap: (bool) If true the bootstrap version is utilized
        :param k_top: (float) K top percent of the samples are used
        :param label_smoothing: (float) Label smoothing factor to be applied
        :param w_0: (float) Weight for label 0
        :param w_1: (float) Weight for label 1
        """
        # Call super constructor
        super(BinaryCrossEntropyLoss, self).__init__()
        self.bootstrap = bootstrap
        self.k_top = k_top
        self.w_0 = w_0
        self.w_1 = w_1
        # Init label smoothing
        self.label_smoothing = None if label_smoothing <= 0.0 else LabelSmoothing(label_smoothing=label_smoothing)

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return self.__class__.__name__

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the binary cross entropy loss of segmentation masks
        :param prediction: (torch.Tensor) Prediction probability
        :param label: (torch.Tensor) Label one-hot encoded
        :return: (torch.Tensor) Loss value
        """
        # Perform label smoothing if utilized
        if self.label_smoothing is not None:
            label = self.label_smoothing(label, 1)
        # Calc binary cross entropy loss
        loss = -(self.w_0 * label * torch.log(prediction.clamp(min=1e-06, max=1. - 1e-06))
                 + self.w_1 * (1.0 - label) * torch.log((1. - prediction.clamp(min=1e-06, max=1. - 1e-06))))
        # Perform bootstrapping
        if self.bootstrap:
            # Flatten loss values
            loss = loss.view(-1)
            # Sort loss values and get k top elements
            loss = loss.sort(descending=True)[0][:int(loss.shape[0] * self.k_top)]
        return loss.mean()


class CrossEntropyLoss(nn.Module):
    """
    This class implements the multi class cross entropy loss.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(CrossEntropyLoss, self).__init__()

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return self.__class__.__name__

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the binary cross entropy loss of segmentation masks
        :param prediction: (torch.Tensor) Prediction probability
        :param label: (torch.Tensor) Label one-hot encoded
        :return: (torch.Tensor) Loss value
        """
        # Calc multi class cross entropy loss
        loss = - (label * torch.log(prediction.clamp(min=1e-06, max=1. - 1e-06))).sum(dim=0)
        return loss.mean()


class BinaryFocalLoss(nn.Module):
    """
    This class implements the segmentation focal loss.
    Paper: https://arxiv.org/abs/1708.02002
    Source: https://github.com/ChristophReich1996/Cell-DETR/blob/master/lossfunction.py
    """

    def __init__(self, gamma: float = 2.0, bootstrap: bool = False, k_top: float = 0.5) -> None:
        """
        Constructor method
        :param gamma: (float) Gamma constant (see paper)
        :param bootstrap: (bool) If true the bootstrap version is utilized
        :param k_top: (float) K top percent of the samples are used
        """
        # Call super constructor
        super(BinaryFocalLoss, self).__init__()
        # Save parameters
        self.gamma = gamma
        self.bootstrap = bootstrap
        self.k_top = k_top

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, gamma={}".format(self.__class__.__name__, self.gamma)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the binary cross entropy loss of segmentation masks
        :param prediction: (torch.Tensor) Prediction probability
        :param label: (torch.Tensor) Label one-hot encoded
        :return: (torch.Tensor) Loss value
        """
        # Calc binary cross entropy loss
        binary_cross_entropy_loss = -(label * torch.log(prediction.clamp(min=1e-06, max=1. - 1e-06))
                                      + (1.0 - label) * torch.log((1.0 - prediction).clamp(min=1e-06, max=1. - 1e-06)))
        # Calc focal loss factor based on the label and the prediction
        focal_factor = prediction * label + (1.0 - prediction) * (1.0 - label)
        # Calc final focal loss
        loss = ((1.0 - focal_factor) ** self.gamma * binary_cross_entropy_loss)
        # Perform bootstrapping
        if self.bootstrap:
            # Flatten loss values
            loss = loss.view(-1)
            # Sort loss values and get k top elements
            loss = loss.sort(descending=True)[0][:int(loss.shape[0] * self.k_top)]
        # Perform reduction
        loss = loss.mean()
        return loss


class FocalLoss(nn.Module):
    """
    Implementation of the multi class focal loss.
    Paper: https://arxiv.org/abs/1708.02002
    Source: https://github.com/ChristophReich1996/Cell-DETR/blob/master/lossfunction.py
    """

    def __init__(self, gamma: float = 2.) -> None:
        """
        Constructor method
        :param gamma: (float) Gamma constant (see paper)
        """
        # Call super constructor
        super(FocalLoss, self).__init__()
        # Save parameters
        self.gamma = gamma

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, gamma={}".format(self.__class__.__name__, self.gamma)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the binary cross entropy loss of segmentation masks
        :param prediction: (torch.Tensor) Prediction probability
        :param label: (torch.Tensor) Label one-hot encoded
        :return: (torch.Tensor) Loss value
        """
        # Calc binary cross entropy loss
        cross_entropy_loss = - (label * torch.log(prediction.clamp(min=1e-06, max=1. - 1e-06))).sum(dim=0)
        # Calc focal loss factor based on the label and the prediction
        focal_factor = (prediction * label + (1.0 - prediction) * (1.0 - label))
        # Calc final focal loss
        loss = ((1.0 - focal_factor) ** self.gamma * cross_entropy_loss).mean()
        return loss


class LabelSmoothing(nn.Module):
    """
    This class implements one-hot label smoothing for Dirichlet segmentation loss
    """

    def __init__(self, label_smoothing: float = 0.05) -> None:
        """
        Constructor method
        :param label_smoothing: (float) Lab-el smoothing factor
        """
        # Call super constructor
        super(LabelSmoothing, self).__init__()
        # Save parameters
        self.label_smoothing = label_smoothing

    def forward(self, label: torch.Tensor, number_of_classes: int) -> torch.Tensor:
        """
        Forward pass smooths a given label
        :param label: (torch.Tensor) Label
        :param number_of_classes: (torch.Tensor) Number of classes
        :return: (torch.Tensor) Smoothed one-hot label
        """
        smooth_positive = 1.0 - self.label_smoothing
        smooth_negative = self.label_smoothing / number_of_classes
        return label * smooth_positive + smooth_negative
