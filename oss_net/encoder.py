from typing import Type, Tuple, Union, Optional, Any

import torch
import torch.nn as nn


class ConvolutionalEncoder(nn.Module):
    """
    This class implements a 3d convolutional encoder for MRI volumes.
    """

    def __init__(self, channels: Tuple[Tuple[int, int], ...] = ((4, 4), (4, 4), (4, 8), (8, 16), (16, 32)),
                 activation: Type[nn.Module] = nn.ReLU,
                 normalization: Type[nn.Module] = nn.BatchNorm3d,
                 dropout: float = 0.0,
                 **kwargs: Any) -> None:
        """
        Constructor method
        :param channels: (Tuple[Tuple[int, int], ...]) Channels (in and out) utilized in each block
        :param activation: (Type[nn.Module]) Type of activation to be used
        :param normalization: (Type[nn.Module]) Type of normalization to be used
        :param dropout: (float) Dropout rate to be utilized
        :param output_channels: (int) Number of output channels to be utilized
        :param kwargs: Key word arguments (not used)
        """
        # Call super constructor
        super(ConvolutionalEncoder, self).__init__()
        # Init mapping
        self.blocks = nn.ModuleList(
            [Residual3dBlock(in_channels=channel[0], out_channels=channel[1], activation=activation,
                             normalization=normalization, downscale=True, dropout=dropout) for channel in channels])
        # Init final mapping
        self.final_mapping = nn.Sequential(
            nn.Conv3d(in_channels=channels[-1][-1], out_channels=1, kernel_size=(5, 5, 5), padding=(2, 2, 2),
                      stride=(1, 1, 1), bias=True),
            normalization(num_features=1, track_running_stats=True, affine=True),
            activation()
        )

    def forward(self, input_volume: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input_volume: (torch.Tensor) Input volume of the shape [batch size, in channels, h, w, d]
        :return: (torch.Tensor) Encoded output tensor of the shape [batch size, 1, (h / n) * (w / n) * (d / n)]
        """
        # Forward pass of all blocks
        for index, block in enumerate(self.blocks):
            input_volume = block(input_volume)
        # Perform output mapping
        output = self.final_mapping(input_volume)
        # Perform segmentation mapping
        return output.flatten(start_dim=2)


class ConvolutionalEncoderSkip(nn.Module):
    """
    This class implements a 3d convolutional encoder for MRI volumes.
    """

    def __init__(self, channels: Tuple[Tuple[int, int], ...] = ((4, 4), (4, 4), (4, 8), (8, 16), (16, 32)),
                 activation: Type[nn.Module] = nn.ReLU,
                 normalization: Type[nn.Module] = nn.BatchNorm3d,
                 dropout: float = 0.0, output_channels: Optional[int] = 6,
                 output_shape: Tuple[int, int, int] = (8, 8, 5),
                 segmentation_mapping: bool = False,
                 **kwargs: Any) -> None:
        """
        Constructor method
        :param channels: (Tuple[Tuple[int, int], ...]) Channels (in and out) utilized in each block
        :param activation: (Type[nn.Module]) Type of activation to be used
        :param normalization: (Type[nn.Module]) Type of normalization to be used
        :param dropout: (float) Dropout rate to be utilized
        :param output_channels: (int) Number of output channels to be utilized
        :param output_shape: (Tuple[int, int, int]) Output volume shape of the encoder
        :param segmentation_mapping: (bool) If true low res. segmentation is predicted form last latent vector
        :param kwargs: Key word arguments (not used)
        """
        # Call super constructor
        super(ConvolutionalEncoderSkip, self).__init__()
        # Init mapping
        self.blocks = nn.ModuleList(
            [Residual3dBlock(in_channels=channel[0], out_channels=channel[1], activation=activation,
                             normalization=normalization, downscale=True, dropout=dropout) for channel in channels])
        # Init output mappings
        self.output_mappings = nn.ModuleList([(nn.Sequential(
            nn.AdaptiveMaxPool3d(output_size=output_shape),
            nn.Conv3d(in_channels=channel[1], out_channels=channel[1] // 2, kernel_size=(5, 5, 5), padding=(2, 2, 2),
                      stride=(1, 1, 1), bias=False),
            normalization(num_features=channel[1] // 2, track_running_stats=True, affine=True),
            activation())) for channel in channels])
        # Init final mapping
        self.final_mapping = nn.Sequential(
            nn.Conv3d(in_channels=sum([channel[1] // 2 for channel in channels]), out_channels=output_channels,
                      kernel_size=(5, 5, 5), padding=(2, 2, 2), stride=(1, 1, 1), bias=True),
            normalization(num_features=output_channels, track_running_stats=True, affine=True),
            activation()
        )
        # Init segmentation mapping
        self.segmentation_mapping = nn.Sequential(
            nn.Conv3d(in_channels=channels[-1][-1], out_channels=1, kernel_size=(5, 5, 5), padding=(2, 2, 2),
                      stride=(1, 1, 1), bias=False),
            nn.Sigmoid()
        ) if segmentation_mapping else None

    def forward(self, input_volume: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass
        :param input_volume: (torch.Tensor) Input volume of the shape [batch size, in channels, h, w, d]
        :return: (torch.Tensor) Encoded output tensor of the shape [batch size, 1, (h / n) * (w / n) * (d / n)]
        """
        outputs = []
        # Forward pass of all blocks
        for index, (block, output_mapping) in enumerate(zip(self.blocks, self.output_mappings)):
            input_volume = block(input_volume)
            outputs.append(output_mapping(input_volume))
        # Concat all output tensors
        features = torch.cat(outputs, dim=1)
        # Perform output mapping
        output = self.final_mapping(features)
        # Perform segmentation mapping
        if self.segmentation_mapping is not None:
            segmentation = self.segmentation_mapping(input_volume)
            return output.flatten(start_dim=2), segmentation
        return output.flatten(start_dim=2)


class Residual3dBlock(nn.Module):
    """
    This class implements a simple 3d residual convolutional block including two convolutions
    """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 normalization: Type[nn.Module] = nn.BatchNorm3d,
                 activation: Type[nn.Module] = nn.ReLU,
                 downscale: bool = True,
                 dropout: float = 0.0) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param normalization: (Type[nn.Module]) Type of normalization to be utilized
        :param activation: (Type[nn.Module]) Type fo activation to be used
        :param downscale: (bool) If true spatial dimensions are downscaled by the factor of two
        :param dropout: (float) Dropout rate to be utilized
        """
        # Call super constructor
        super(Residual3dBlock, self).__init__()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1), bias=False),
            normalization(num_features=out_channels, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)) if downscale else nn.Identity(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1), bias=False),
            normalization(num_features=out_channels, affine=True, track_running_stats=True)
        )
        # Init residual mapping
        self.residual_mapping = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1),
                                          stride=(1, 1, 1), padding=(0, 0, 0),
                                          bias=False) if in_channels != out_channels else nn.Identity()
        # Init final activation
        self.final_activation = activation()
        # Init pooling layer for downscaling the spatial dimensions
        self.pooling_layer = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)) if downscale else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Froward pass
        :param input: (torch.Tensor) Input volume tensor of the shape [batch size, out channels, h, w, d]
        :return: (torch.Tensor) Output volume tensor of the shape [batch size, in  channels, h / 2, w / 2, d / 2]
        """
        # Forward pass main mapping
        output = self.main_mapping(input)
        # Residual mapping
        output = output + self.pooling_layer(self.residual_mapping(input))
        # Perform final activation
        output = self.final_activation(output)
        # Downscale spatial dimensions
        return output
