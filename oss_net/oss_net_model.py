from typing import Dict, List

import torch
import torch.nn as nn

from oss_net.encoder import *
from oss_net.decoder import *


class OSSNet(nn.Module):
    """
    This class implements the occupancy network for semantic segmentation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        # Call super constructor
        super(OSSNet, self).__init__()
        # Init encoder
        self.encoder = config["encoder"](
            channels=config["encoder_channels"],
            normalization=config["encoder_normalization"],
            segmentation_mapping=config["encoder_segmentation_mapping"],
            activation=config["activation"],
            dropout=config["dropout"],
            output_channels=len(config["decoder_features"]) + 1,
            output_shape=config["encoder_output_shape"]
        )
        # Init decoder
        self.decoder = config["decoder"](
            features=config["decoder_features"],
            latent_features=config["latent_features"],
            activation=config["activation"],
            dropout=config["dropout"],
            fourier_input_features=config["decoder_fourier_input_features"],
            patch_mapping=config["decoder_patch_mapping"],
            large_patches=config["large_patches"],
            output_feautres=config["number_of_classes"],
            patch_channels=config["encoder_channels"][0][0]
        )

    def parameter_dicts(self, lr: float = 1e-03, lr_pau: float = 1e-02,
                        weight_decay_encoder: float = 0.,
                        weight_decay_decoder: float = 0.) -> List[Dict[str, Union[torch.Tensor, float]]]:
        """

        :param lr: (float) Learning rate of all parameters except for PAU
        :param lr_pau: (float) PAU learning rate
        :param weight_decay_encoder: (float) Weight decay applied in encoder
        :param weight_decay_decoder: (float) Weight decay applied in decoder
        :return: (List[Dict[str, Union[torch.Tensor, float]]]) List of parameter dicts
        """
        parameter = []
        for key, value in self.named_parameters():
            if value.is_leaf:
                if ("numerator" in key) or ("denominator" in key):
                    parameter.append({"params": value, "lr": lr_pau, "weight_decay": 0.})
                elif "encoder" in key:
                    parameter.append({"params": value, "lr": lr, "weight_decay": weight_decay_encoder})
                elif "decoder" in key:
                    parameter.append({"params": value, "lr": lr, "weight_decay": weight_decay_decoder})
                else:
                    parameter.append({"params": value, "lr": lr, "weight_decay": 0.})
        return parameter

    def forward(self, volumes: torch.Tensor,
                coordinates: torch.Tensor,
                patches: Optional[torch.Tensor],
                inference: bool = False,
                latent_vectors: Optional[torch.Tensor] = None,
                max_coordinates: Optional[int] = 2 ** 14) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        :param volumes: (torch.Tensor) Input volumes of the shape [batch size, channels (4), h, w, d]
        :param coordinates: (torch.Tensor) Coordinates of the shape [batch size, n coordinates, 3]
        :param patches: (Optional[torch.Tensor]) Patches
        [batch size, n coordinates, channels (4), h * w * d (5 * 5 * 5)]
        :param inference: (bool) If true inference mode is utilized
        :param latent_vectors: (Optional[torch.Tensor]) Latent vector
        :param max_coordinates: (Optional[int]) Max coordinates to process per forward pass
        :return: (torch.Tensor) Occupancy probability vector corresponding to the semantic class with the shape
                [batch size, n coordinates, n classes]
        """
        # Save shape of coordinates
        coordinates_shape = coordinates.shape
        # Make tuple of coordinates and patches of max number of coordinates is reached
        if max_coordinates is not None:
            if coordinates.shape[1] > max_coordinates and inference:
                coordinates = coordinates.split(max_coordinates, dim=1)
                if patches is not None:
                    patches = patches.split(max_coordinates, dim=1)
                else:
                    patches = None
        # Encoder volumes
        if latent_vectors is None or not inference:
            output = self.encoder(volumes)
            if isinstance(output, tuple):
                latent_vectors, segmentation_low_res = output
            else:
                latent_vectors = output
                segmentation_low_res = None
        # Predict occupancy segmentation
        if patches is not None:
            if isinstance(coordinates, tuple):
                output = torch.cat([self.decoder(coordinate.reshape(-1, 1, 3),
                                                 patch.flatten(start_dim=0, end_dim=1),
                                                 latent_vectors)
                                    for coordinate, patch in zip(coordinates, patches)], dim=0)
            else:
                output = self.decoder(coordinates.reshape(-1, 1, 3),
                                      patches.flatten(start_dim=0, end_dim=1),
                                      latent_vectors)
        else:
            if isinstance(coordinates, tuple):
                output = torch.cat([self.decoder(coordinate.reshape(-1, 1, 3),
                                                 None,
                                                 latent_vectors)
                                    for coordinate in coordinates], dim=1)
            else:
                output = self.decoder(coordinates.reshape(-1, 1, 3),
                                      None,
                                      latent_vectors)
        # Reshape output to [batch size, n coordinates, number of classes]
        output = output.reshape(coordinates_shape[0], coordinates_shape[1], output.shape[-1])
        if inference:
            return output, latent_vectors
        return output, segmentation_low_res

    @torch.no_grad()
    def inference(self, volume: torch.Tensor, patches: torch.Tensor, resolution: Tuple[int, int, int],
                  resolution_stages: Tuple[int, ...] = (4, 2, 1), threshold: float = 0.5,
                  large_patches: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method implements the OSS-Net inference algorithm to extract a dense voxelized prediction.
        :param volume: (torch.Tensor) Downscaled input volume
        :param patches: (torch.Tensor) Small input patches
        :param resolution: (Tuple[int, int, int]) Resolution of the dense segmentation to be predicted
        :param resolution_stages: (Tuple[int, ...]) Resolution stages to be evaluated
        :param threshold: (float) Segmentation threshold
        :param large_patches: (Optional[torch.Tensor]) Large input patches
        :return: (torch.Tensor) Dense segmentation prediction
        """
        # Check batch size
        assert volume.shape[0] == 1, "Only a batch size of one is supported!"
        # Init occupancy grid
        occupancy_grid = torch.zeros(resolution, dtype=torch.float, device=volume.device)
        # Perform different resolution stages
        for index, resolution_stage in enumerate(resolution_stages):
            if index == 0:
                # Init coordinates
                coordinates = torch.stack(
                    torch.meshgrid(torch.arange(0, resolution[0], resolution_stage),
                                   torch.arange(0, resolution[1], resolution_stage),
                                   torch.arange(0, resolution[2], resolution_stage)), dim=-1).reshape(-1, 3)
                # Get input patches if utilized
                if patches is not None:
                    input_patches = patches[:, :,
                                    coordinates[:, 0],
                                    coordinates[:, 1],
                                    coordinates[:, 2]].transpose(1, 2).float()
                    # Get large patches if utilized
                    if large_patches is not None:
                        large_input_patches = large_patches[:, :,
                                              coordinates[:, 0],
                                              coordinates[:, 1],
                                              coordinates[:, 2]].transpose(1, 2).float()
                        # Downscale patches
                        large_input_patches = F.max_pool3d(large_input_patches[0], kernel_size=(2, 2, 2),
                                                           stride=(2, 2, 2))[None]
                        # Concat small and large patches
                        input_patches = torch.cat([input_patches, large_input_patches], dim=2)
                    # Data to device
                    input_patches = input_patches.to(volume.device)
                else:
                    input_patches = None
                coordinates = coordinates.to(volume.device)
                # Add batch dimension to coordinates
                coordinates = coordinates.unsqueeze(dim=0)
                # Make prediction
                occupancy_prediction, latent_vector = self.forward(volume, coordinates, input_patches,
                                                                   inference=True)
                # Get coordinates of detected position
                coordinates = coordinates[0][occupancy_prediction[0, :, 0] > threshold]
                # Add predictions to occupancy grid
                occupancy_grid[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1.0
            else:
                # Init coordinates
                coordinates = torch.stack(
                    torch.meshgrid(torch.arange(0, resolution[0], resolution_stage),
                                   torch.arange(0, resolution[1], resolution_stage),
                                   torch.arange(0, resolution[2], resolution_stage)), dim=-1).reshape(-1, 3)
                # Data to device
                coordinates = coordinates.to(volume.device)
                # Erase occupancy grid to get coordinates for next iteration
                occupancy_grid_erased = F.max_pool3d(
                    occupancy_grid[None, None],
                    kernel_size=(3 ** resolution_stage, 3 ** resolution_stage, 3 ** resolution_stage),
                    stride=(1, 1, 1),
                    padding=(
                        3 ** resolution_stage // 2, 3 ** resolution_stage // 2, 3 ** resolution_stage // 2))[
                    0, 0]
                # Get possible positions to evaluate
                possible_positions = (occupancy_grid_erased != occupancy_grid)[
                    coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]
                # Get coordinates to evaluate
                coordinates = coordinates[possible_positions == 1.]
                # Get input patches if utilized
                if patches is not None:
                    input_patches = patches[:, :,
                                    coordinates[:, 0],
                                    coordinates[:, 1],
                                    coordinates[:, 2]].transpose(1, 2).float()
                    # Get large patches if utilized
                    if large_patches is not None:
                        large_input_patches = large_patches[:, :,
                                              coordinates[:, 0],
                                              coordinates[:, 1],
                                              coordinates[:, 2]].transpose(1, 2).float()
                        # Downscale patches
                        large_input_patches = F.max_pool3d(large_input_patches[0], kernel_size=(2, 2, 2),
                                                           stride=(2, 2, 2))[None]
                        # Concat small and large patches
                        input_patches = torch.cat([input_patches, large_input_patches], dim=2)
                    input_patches = input_patches.to(volume.device)
                # Add batch dimension to coordinates
                coordinates = coordinates.unsqueeze(dim=0)
                # Make prediction
                occupancy_prediction, latent_vector = self.forward(None, coordinates, input_patches,
                                                                   inference=True, latent_vectors=latent_vector)
                # Get coordinates of detected position
                coordinates = coordinates[0][occupancy_prediction[0, :, 0] > threshold]
                # Add predictions to occupancy grid
                occupancy_grid[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1.0
        return occupancy_grid, latent_vector

    @torch.no_grad()
    def inference_optimized(self, volume: torch.Tensor, patches: torch.Tensor, resolution: Tuple[int, int, int],
                            resolution_stages: Tuple[int, ...] = (4, 2, 1), threshold: float = 0.5,
                            large_patches: Optional[torch.Tensor] = None,
                            threshold_low_res_seg: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method implements the OSS-Net inference algorithm to extract a dense voxelized prediction.
        :param volume: (torch.Tensor) Downscaled input volume
        :param patches: (torch.Tensor) Small input patches
        :param resolution: (Tuple[int, int, int]) Resolution of the dense segmentation to be predicted
        :param resolution_stages: (Tuple[int, ...]) Resolution stages to be evaluated
        :param threshold: (float) Segmentation threshold
        :param large_patches: (torch.Tensor) Large input patches
        :return: (torch.Tensor) Dense segmentation prediction
        """
        # Check batch size
        assert volume.shape[0] == 1, "Only a batch size of one is supported!"
        # Init occupancy grid
        occupancy_grid = torch.zeros(resolution, dtype=torch.float, device=volume.device)
        # Encode volume to produce the latent vector and low resolution segmentation
        latent_vector, segmentation_low_res = self.encoder(volume)
        segmentation_low_res = F.interpolate(segmentation_low_res, size=(resolution[0] // resolution_stages[0],
                                                                         resolution[1] // resolution_stages[0],
                                                                         resolution[2] // resolution_stages[0]),
                                             mode="trilinear", align_corners=True)
        segmentation_low_res = (segmentation_low_res > threshold_low_res_seg)
        # Perform different resolution stages
        for index, resolution_stage in enumerate(resolution_stages):
            if index == 0:
                # Init coordinates
                coordinates = torch.stack(
                    torch.meshgrid(torch.arange(0, resolution[0], resolution_stage),
                                   torch.arange(0, resolution[1], resolution_stage),
                                   torch.arange(0, resolution[2], resolution_stage)), dim=-1).reshape(-1, 3)
                # Get coordinates to evaluate
                coordinates = coordinates[segmentation_low_res.view(-1) == 1.]
                # Get input patches if utilized
                if patches is not None:
                    input_patches = patches[:, :,
                                    coordinates[:, 0],
                                    coordinates[:, 1],
                                    coordinates[:, 2]].transpose(1, 2).float()
                    # Get large patches if utilized
                    if large_patches is not None:
                        large_input_patches = large_patches[:, :,
                                              coordinates[:, 0],
                                              coordinates[:, 1],
                                              coordinates[:, 2]].transpose(1, 2).float()
                        # Downscale patches
                        large_input_patches = F.max_pool3d(large_input_patches[0], kernel_size=(2, 2, 2),
                                                           stride=(2, 2, 2))[None]
                        # Concat small and large patches
                        input_patches = torch.cat([input_patches, large_input_patches], dim=2)
                    # Data to device
                    input_patches = input_patches.to(volume.device)
                else:
                    input_patches = None
                coordinates = coordinates.to(volume.device)
                # Add batch dimension to coordinates
                coordinates = coordinates.unsqueeze(dim=0)
                # Make prediction
                occupancy_prediction, latent_vector = self.forward(None, coordinates, input_patches,
                                                                   inference=True, latent_vectors=latent_vector)
                # Get coordinates of detected position
                coordinates = coordinates[0][occupancy_prediction[0, :, 0] > threshold]
                # Add predictions to occupancy grid
                occupancy_grid[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1.0
            else:
                # Init coordinates
                coordinates = torch.stack(
                    torch.meshgrid(torch.arange(0, resolution[0], resolution_stage),
                                   torch.arange(0, resolution[1], resolution_stage),
                                   torch.arange(0, resolution[2], resolution_stage)), dim=-1).reshape(-1, 3)
                # Data to device
                coordinates = coordinates.to(volume.device)
                # Erase occupancy grid to get coordinates for next iteration
                occupancy_grid_erased = F.max_pool3d(
                    occupancy_grid[None, None],
                    kernel_size=(3 ** resolution_stage, 3 ** resolution_stage, 3 ** resolution_stage),
                    stride=(1, 1, 1),
                    padding=(3 ** resolution_stage // 2, 3 ** resolution_stage // 2, 3 ** resolution_stage // 2))[0, 0]
                # Get possible positions to evaluate
                possible_positions = (occupancy_grid_erased != occupancy_grid)[
                    coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]
                # Get coordinates to evaluate
                coordinates = coordinates[possible_positions == 1.]
                # Get input patches if utilized
                if patches is not None:
                    input_patches = patches[:, :,
                                    coordinates[:, 0],
                                    coordinates[:, 1],
                                    coordinates[:, 2]].transpose(1, 2).float()
                    # Get large patches if utilized
                    if large_patches is not None:
                        large_input_patches = large_patches[:, :,
                                              coordinates[:, 0],
                                              coordinates[:, 1],
                                              coordinates[:, 2]].transpose(1, 2).float()
                        # Downscale patches
                        large_input_patches = F.max_pool3d(large_input_patches[0], kernel_size=(2, 2, 2),
                                                           stride=(2, 2, 2))[None]
                        # Concat small and large patches
                        input_patches = torch.cat([input_patches, large_input_patches], dim=2)
                    input_patches = input_patches.to(volume.device)
                # Add batch dimension to coordinates
                coordinates = coordinates.unsqueeze(dim=0)
                # Make prediction
                occupancy_prediction, latent_vector = self.forward(None, coordinates, input_patches,
                                                                   inference=True, latent_vectors=latent_vector)
                # Get coordinates of detected position
                coordinates = coordinates[0][occupancy_prediction[0, :, 0] > threshold]
                # Add predictions to occupancy grid
                occupancy_grid[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1.0
        return occupancy_grid, latent_vector
