from typing import Type, Tuple, Any
import math

import torch
import torch.nn as nn

from pade_activation_unit.utils import PAU


class ResidualCBNDecoder(nn.Module):
    """
    This class implements a residual and Conditional Batch Normalization (CBN) based occupancy decoder.
    """

    def __init__(self,
                 output_features: int = 1,
                 latent_features: int = 320,
                 features: Tuple[Tuple[int, int], ...] = (
                         (448, 256), (256, 256), (256, 256), (256, 256), (256, 256)),
                 activation: Type[nn.Module] = nn.ReLU,
                 dropout: float = 0.0,
                 fourier_input_features: bool = True,
                 patch_mapping: bool = True,
                 large_patches: bool = True,
                 patch_channels: int = 4,
                 **kwargs: Any) -> None:
        """
        Constructor method
        :param output_features: (int) Number of output features (binary classification = 1)
        :param latent_features: (int) Number of features in the encoder latent vector
        :param features: (Tuple[Tuple[int, int], ...]) Features (in and out) utilized in each block
        :param activation: (Type[nn.Module]) Type of activation function to be utilized
        :param dropout: (float) Dropout rate to be utilized
        :param fourier_input_features: (bool) If true random fourier input features are utilized
        :param patch_mapping: (bool) If true patch mapping and patches are utilized
        :param large_patches: (bool) If true additional channels are used in patch mapping for large patches
        :param kwargs: Key word arguments (not used)
        """
        # Call super constructor
        super(ResidualCBNDecoder, self).__init__()
        # Save parameters
        self.fourier_input_features = fourier_input_features
        if self.fourier_input_features:
            self.register_buffer("b", torch.randn(1, 3, 16) * 4.)
        # Init residual blocks
        self.blocks = nn.ModuleList(
            [ResidualCBNFFNNBlock(in_features=feature[0], out_features=feature[1], latent_features=latent_features,
                                  activation=activation, dropout=dropout) for feature in features])
        # Init patch mapping
        self.patch_mapping = PatchMapping(in_channels=patch_channels * 2 if large_patches else patch_channels,
                                          out_channels=1, activation=activation,
                                          dropout=dropout) if patch_mapping else None
        # Init final layer and activation
        self.final_mapping = nn.Sequential(
            nn.Linear(in_features=features[-1][-1], out_features=output_features, bias=False),
            nn.Softmax(dim=-1) if output_features > 1 else nn.Sigmoid()
        )

    def forward(self, coordinates: torch.Tensor, patches: torch.Tensor, latent_vectors: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param coordinates: (torch.Tensor) Input coordinates
        :param patches: (torch.Tensor) Patches
        :param latent_vectors: (torch.Tensor) Latent vectors
        :return: (torch.Tensor) Output tensor
        """
        # Perform fourier feature mapping if utilized
        if self.fourier_input_features:
            coordinates = torch.cat([torch.cos(2 * math.pi * coordinates @ self.b),
                                     torch.sin(2 * math.pi * coordinates @ self.b),
                                     coordinates], dim=-1)
        # Construct input tensor
        if self.patch_mapping is not None:
            input = torch.cat([
                coordinates,
                self.patch_mapping(patches).flatten(start_dim=2),
                latent_vectors[:, :1].repeat_interleave(repeats=coordinates.shape[0] // latent_vectors.shape[0],
                                                        dim=0)],
                dim=-1)
        else:
            input = torch.cat([
                coordinates,
                latent_vectors[:, :1].repeat_interleave(repeats=coordinates.shape[0] // latent_vectors.shape[0],
                                                        dim=0)],
                dim=-1)
        # Forward pass residual CBN blocks
        if latent_vectors.shape[1] == 1:
            for index, block in enumerate(self.blocks):
                input = block(input, latent_vectors)
        else:
            for index, block in enumerate(self.blocks):
                input = block(input, latent_vectors[:, index + 1:index + 2])
        # Forward pass final layer and activation
        output = self.final_mapping(input).squeeze(dim=1)
        return output


class ResidualCBNFFNNBlock(nn.Module):
    """
    This class implements a simple residual feed-forward neural network block with two linear layers and CBN.
    """

    def __init__(self, in_features: int, out_features: int, latent_features: int,
                 activation: Type[nn.Module] = PAU, dropout: float = 0.0) -> None:
        """
        Constructor method
        :param in_features: (int) Number of input features
        :param out_features: (int) Number of output features
        :param latent_features: (int) Number of latent features
        :param activation: (Type[nn.Module]) Type of activation function to be utilized
        :param dropout: (float) Dropout rate to be utilized
        """
        # Call super constructor
        super(ResidualCBNFFNNBlock, self).__init__()
        # Init linear layers
        self.linear_layer_1 = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        self.linear_layer_2 = nn.Linear(in_features=out_features, out_features=out_features, bias=True)
        # Init activations
        self.activation = activation()
        self.final_activation = activation()
        # Init dropout layer
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        # Init residual mapping
        self.residual_mapping = nn.Linear(in_features=in_features, out_features=out_features,
                                          bias=True) if in_features != out_features else nn.Identity()
        # Init CNB
        self.cnb_1 = ConditionalBatchNorm1d(latent_features=latent_features, input_features=out_features)
        self.cnb_2 = ConditionalBatchNorm1d(latent_features=latent_features, input_features=out_features)

    def forward(self, input: torch.Tensor, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size * coordinates, *, in features]
        :param latent_vector: (torch.Tensor) Encoded latent vector of the shape [batch size, latent features]
        :return: (torch.Tensor) Output tensor of the shape [batch size * coordinates, *, out features]
        """
        # Forward pass first stage
        output = self.linear_layer_1(input)
        output = self.cnb_1(output, latent_vector)
        output = self.activation(output)
        output = self.dropout(output)
        # Forward pass second stage
        output = self.linear_layer_2(output)
        output = self.cnb_2(output, latent_vector)
        # Forward pass residual mapping and final activation
        output = output + self.residual_mapping(input)
        output = self.final_activation(output)
        return output


class ConditionalBatchNorm1d(nn.Module):
    """
    Implementation of a conditional batch normalization module using linear operation to predict gamma and beta
    """

    def __init__(self, latent_features: int, input_features: int,
                 normalization: Type[nn.Module] = nn.BatchNorm1d) -> None:
        """
        Conditional batch normalization module including two 1D convolutions to predict gamma end beta
        :param latent_features: (int) Features of the latent vector
        :param input_features: (int) Features of the output vector to be normalized
        :param normalization: (Type[nn.Module]) Type of normalization to be utilized
        """
        super(ConditionalBatchNorm1d, self).__init__()
        # Init operations
        self.linear_gamma = nn.Linear(in_features=latent_features, out_features=input_features, bias=True)
        self.linear_beta = nn.Linear(in_features=latent_features, out_features=input_features, bias=True)
        self.normalization = normalization(num_features=1, affine=False, track_running_stats=True, momentum=0.1)
        # Reset parameters of convolutions
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Method resets the parameter of the convolution to predict gamma and beta
        """
        nn.init.zeros_(self.linear_gamma.weight)
        nn.init.zeros_(self.linear_beta.weight)
        nn.init.ones_(self.linear_gamma.bias)
        nn.init.zeros_(self.linear_beta.bias)

    def forward(self, input: torch.Tensor, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor to be normalized of shape (batch size * coordinates, 1, features)
        :param latent_vector: (torch.Tensor) Latent vector tensor of shape (batch_size, features)
        :return: (torch.Tensor) Normalized tensor
        """
        # Perform linear layers to estimate gamma and beta
        gamma = self.linear_gamma(latent_vector)
        beta = self.linear_beta(latent_vector)
        # Perform normalization
        output_normalized = self.normalization(input)
        # Repeat gamma and beta to apply factors to every coordinate
        gamma = gamma.repeat_interleave(output_normalized.shape[0] // gamma.shape[0], dim=0)
        beta = beta.repeat_interleave(output_normalized.shape[0] // beta.shape[0], dim=0)
        # Add factors
        output = gamma * output_normalized + beta
        return output


class PatchMapping(nn.Module):
    """
    This class implements a patch mapping module.
    """

    def __init__(self, in_channels: int, out_channels: int, activation: Type[nn.Module] = nn.PReLU,
                 dropout: float = 0.) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param activation: (Type[nn.Module]) Type of activation to be utilized
        """
        # Call super constructor
        super(PatchMapping, self).__init__()
        # Init mapping
        self.mapping = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=max(in_channels // 2, 1), kernel_size=(7, 7, 7),
                      stride=(1, 1, 1), padding=(3, 3, 3), bias=True),
            activation(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Conv3d(in_channels=max(in_channels // 2, 1), out_channels=out_channels, kernel_size=(7, 7, 7),
                      stride=(1, 1, 1), padding=(3, 3, 3), bias=True),
            activation()
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param patches: (torch.Tensor) Input volume of 3d patches
        :return: (torch.Tensor) Output feature tensor
        """
        output = self.mapping(patches)
        return output
