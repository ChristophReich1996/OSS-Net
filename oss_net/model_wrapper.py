from typing import Union, Optional, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from scipy.ndimage import label
from rtpt.rtpt import RTPT
import mcubes
import open3d as o3d

import metric
import misc
import visualization


class ModelWrapper(object):
    """
    This class implements a wrapper of the AOS-Net, which implements training and testing methods.
    """

    def __init__(self,
                 model: Union[nn.DataParallel, nn.Module],
                 optimizer: torch.optim.Optimizer,
                 loss_function: nn.Module,
                 loss_function_latent_segmentation: nn.Module,
                 training_dataset: DataLoader,
                 test_dataset: DataLoader,
                 learning_rate_schedule: torch.optim.lr_scheduler.MultiStepLR,
                 device: Optional[str] = "cuda",
                 data_logger: Optional[misc.Logger] = None,
                 segmentation_loss_factor: Optional[float] = 0.001) -> None:
        """
        Constructor method
        :param model: (nn.Module) Occupancy model to be trained
        :param optimizer: (torch.optim.Optimizer) Optimizer of the model
        :param loss_function: (nn.Module) Loss function to be utilized
        :param loss_function_latent_segmentation: (nn.Module) Loss function for the latent segmentation
        :param training_dataset: (DataLoader) Training dataset
        :param test_dataset: (DataLoader) Test dataset
        :param learning_rate_schedule: (torch.optim.lr_scheduler.LambdaLR) Learning rate schedule
        :param device: (Optional[str]) Device to be utilized
        :param data_logger: (Optional[misc.Logger]) Data logger object
        :param segmentation_loss_factor: (Optional[float]) Segmentation loss factor to be applied
        """
        # Save parameters
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.loss_function_latent_segmentation = loss_function_latent_segmentation
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.learning_rate_schedule = learning_rate_schedule
        self.device = device
        self.data_logger = data_logger
        self.segmentation_loss_factor = segmentation_loss_factor

    def train(self, epochs: Optional[int] = 20, validate_after_n_epochs: Optional[int] = 1,
              save_model_after_n_epochs: Optional[int] = 1, save_best_model: Optional[bool] = True) -> None:
        """
        Training method of the AOS-Net.
        :param epochs: (Optional[int]) Number of epochs to be performed
        :param validate_after_n_epochs: (Optional[int]) Number of epochs after the model gets evaluated
        :param save_model_after_n_epochs: (Optional[int]) Number of epochs after the model is saved
        :param save_best_model: (Optional[bool]) If true the best model performing model based on validation is saved
        """
        # Model into training mode
        self.model.train()
        # Model to device
        self.model.to(self.device)
        # Init progress bar
        self.progress_bar = tqdm(total=epochs * len(self.training_dataset))
        # Init best validation metric
        best_validation_metric = 0.0
        # Init rtpt
        rtpt = RTPT(name_initials="CR", experiment_name="OSS-Net", max_iterations=epochs)
        # Start rtpt
        rtpt.start()
        # Main training loop
        for self.epoch in range(epochs):
            # Rtpt step
            rtpt.step(subtitle="IoU={:.3f}".format(best_validation_metric))
            for batch in self.training_dataset:
                # Update progress bar
                self.progress_bar.update(n=1)
                # Reset gradients of model
                self.optimizer.zero_grad()
                # Case if dataset does not return patches
                volumes, coordinates, labels, patches, segmentations = batch
                # Data to device
                volumes = volumes.to(self.device)
                coordinates = coordinates.to(self.device)
                labels = labels.to(self.device)
                patches = patches.to(self.device) if patches is not None else patches
                segmentations = segmentations.to(self.device) if segmentations is not None else segmentations
                # Make prediction
                predictions, latent_segmentations = self.model(volumes, coordinates, patches)
                # Calc loss
                loss = self.loss_function(predictions, labels)
                # If segmentation loss is utilized
                if (self.loss_function_latent_segmentation is not None) and (latent_segmentations is not None):
                    # Calc segmentation loss
                    segmentation_loss = self.loss_function_latent_segmentation(latent_segmentations, segmentations)
                    # Sum op losses
                    loss_full = loss + self.segmentation_loss_factor * segmentation_loss
                    # Calc gradients
                    loss_full.backward()
                else:
                    # Calc gradients
                    loss.backward()
                # Optimize model
                self.optimizer.step()
                # Show loss in progress bar
                self.progress_bar.set_description(
                    "Epoch {}/{} L={:.4f} IoU={:.4f}".format(self.epoch + 1, epochs, loss.item(),
                                                             best_validation_metric))
                # Log loss
                self.data_logger.log_metric(metric_name="Loss", value=loss.cpu().item())
            # Update learning rate schedule
            if self.learning_rate_schedule is not None:
                self.learning_rate_schedule.step()
            # Perform validation if utilized
            if ((self.epoch + 1) % validate_after_n_epochs) == 0:
                validation_metric = self.validate(return_metric=True, plot_predictions=True)
                # Ensure train mode
                self.model.train()
                # Check if new best metric is present
                if best_validation_metric < validation_metric:
                    best_validation_metric = validation_metric
                    # Save best model if utilized
                    if save_best_model:
                        self.data_logger.log_model(
                            file_name="best_model.pt",
                            model=self.model.module if isinstance(self.model, nn.DataParallel) else self.model)
            # Save current model
            if ((self.epoch + 1) % save_model_after_n_epochs == 0):
                self.data_logger.log_model(
                    file_name="{}_model.pt".format(self.epoch),
                    model=self.model.module if isinstance(self.model, nn.DataParallel) else self.model)
            # Save logs
            self.data_logger.save()
        # Close progress bar
        self.progress_bar.close()

    @torch.no_grad()
    def validate(self, metrics: Optional[Tuple[nn.Module, ...]] = (metric.IoU(), metric.DiceScore()),
                 return_metric: Optional[bool] = False, metric_to_return: Optional[str] = str(metric.IoU()),
                 plot_predictions: Optional[bool] = False) -> Union[None, float]:
        """
        Validation method
        :param metrics: (Optional[Tuple[nn.Module, ...]]) Test metrics
        :param return_metric: (Optional[bool]) If true computed metric is returned
        :param metric_to_return: (Optional[str]) Metric to be returned
        :param plot_predictions: (Optional[bool]) If true plot of the predictions will be produced
        :return: (Union[None, float]) Computed metric if utilized else None
        """
        # Model into eval mode
        self.model.eval()
        # Model to device
        self.model.to(self.device)
        # Testing loop
        for index, batch in enumerate(self.test_dataset):
            # Set progress bar description if progress bar is present
            try:
                self.progress_bar.set_description("Testing {}/{}".format(index + 1, len(self.test_dataset)))
            except AttributeError:
                pass
            # Case if dataset does not return patches
            volumes, coordinates, labels, patches, segmentations = batch
            # Data to device
            volumes = volumes.to(self.device)
            coordinates = coordinates.to(self.device)
            labels = labels.to(self.device)
            patches = patches.to(self.device) if patches is not None else patches
            del segmentations
            # Make prediction
            predictions, _ = self.model(volumes, coordinates, patches)
            # Calc and log loss
            self.data_logger.log_temp_metric(metric_name="Test_Loss", value=self.loss_function(predictions, labels))
            # Calc all metrics
            for test_metric in metrics:
                self.data_logger.log_temp_metric(metric_name=str(test_metric), value=test_metric(predictions, labels))
            # Plot prediction and label if utilized
            if plot_predictions:
                # Set prediction offset
                predictions_offset = (predictions > 0.5).float()
                # Denormalize coordinates
                coordinates_denormalized = self.training_dataset.dataset.denormalize_coordinates(coordinates)
                try:
                    visualization.make_pointcloud_obj(
                        coordinates_denormalized[0][predictions_offset[0, :, 0] == 1.0],
                        coordinates_denormalized[0][labels[0, :, 0] == 1.0],
                        volumes, side_len=1,
                        path=os.path.join(self.data_logger.path_plots, "{}_{}".format(self.epoch, index)))
                except AttributeError:
                    visualization.make_pointcloud_obj(
                        coordinates_denormalized[0][predictions_offset[0, :, 0] == 1.0],
                        coordinates_denormalized[0][labels[0, :, 0] == 1.0],
                        volumes, side_len=1,
                        path=os.path.join(self.data_logger.path_plots, "test_{}".format(index)))
        # Average metrics
        metric_results = self.data_logger.save_temp_metric(
            metric_name=["Test_Loss"] + [str(test_metric) for test_metric in metrics])
        # Return defined metric if utilized
        if return_metric:
            return metric_results[metric_to_return]

    def inference(self, inference_dataset: Dataset, resolution_stages: Optional[Tuple[int]] = (4, 2, 1),
                  threshold: Optional[float] = 0.5,
                  metrics: Optional[Tuple[nn.Module, ...]] = (metric.IoU(), metric.DiceScore()), ) -> None:
        """
        This method performs inference with the occupancy network and save the resulting mesh as obj file
        :param inference_dataset: (DataLoader) Infer. dataset which returns the input volume and patches (only bs 1!)
        :param starting_resolution: (Optional[int]) Fraction of the original resolution to start occupancy evaluation
        :param threshold: (Optional[float]) Threshold applied to prediction
        """
        # Model into eval mode
        self.model.eval()
        # Model to device
        self.model.to(self.device)
        # Init progress bar
        progress_bar = tqdm(total=len(inference_dataset))
        for batch_index, (volume, patches, label, large_patches) in enumerate(inference_dataset):
            # Update progress bar
            progress_bar.update(n=1)
            # Volume to device
            volume = volume.to(self.device)
            # Get dense prediction with standard inference
            inference_method = self.model.module.inference \
                if isinstance(self.model, nn.DataParallel) else self.model.inference
            # Start timing
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            occupancy_grid_standard, latent_vector = inference_method(volume=volume, patches=patches,
                                                                      resolution=inference_dataset.padding_shape,
                                                                      resolution_stages=resolution_stages,
                                                                      threshold=threshold, large_patches=large_patches)
            # End timing
            end.record()
            torch.cuda.synchronize()
            runtime = start.elapsed_time(end)
            # Log runtime
            self.data_logger.log_metric(metric_name="Runtime", value=float(runtime))
            # Get dense prediction
            inference_method = self.model.module.inference_optimized \
                if isinstance(self.model, nn.DataParallel) else self.model.inference_optimized
            # Start timing
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            occupancy_grid, latent_vector = inference_method(volume=volume, patches=patches,
                                                             resolution=inference_dataset.padding_shape,
                                                             resolution_stages=resolution_stages,
                                                             threshold=threshold, large_patches=large_patches)
            # End timing
            end.record()
            torch.cuda.synchronize()
            runtime = start.elapsed_time(end)
            # Log runtime
            self.data_logger.log_metric(metric_name="Runtime_optimized", value=float(runtime))
            # Log disagreement between predictions
            self.data_logger.log_metric(metric_name="Disagreement",
                                        value=(occupancy_grid != occupancy_grid_standard).sum().item())
            # Calc metrics
            for test_metric in metrics:
                self.data_logger.log_temp_metric(metric_name=str(test_metric),
                                                 value=test_metric(occupancy_grid.view(1, -1).cpu(),
                                                                   label.view(1, -1).cpu()))
            # Get mesh from prediction
            vertices_prediction, triangles_prediction = self.extract_mash(occupancy_grid, smooth=False,
                                                                          refine=True, latent_vector=latent_vector,
                                                                          simplify=True, patches=patches,
                                                                          large_patches=large_patches)
            # Save occupancy grid
            self.data_logger.save_occupancy_grid(occupancy_grid, "occupancy_grid_{}.pt".format(batch_index))
            # Save mesh
            self.data_logger.save_mesh(vertices=vertices_prediction,
                                       triangles=triangles_prediction,
                                       name=("mesh_vertices_prediction_{}.pt".format(batch_index),
                                             "mesh_triangels_prediction_{}.pt".format(batch_index)))
            mcubes.export_obj(vertices=vertices_prediction.numpy(), triangles=triangles_prediction.numpy(),
                              filename=os.path.join(self.data_logger.path_plots,
                                                    "mesh_prediction_{}.obj".format(batch_index)))
            # Get mesh from label
            vertices_label, triangles_label = self.extract_mash(label, cluster_offset=0, smooth=False, refine=False,
                                                                latent_vector=None, simplify=False, patches=None,
                                                                large_patches=None)
            # Save mesh
            self.data_logger.save_mesh(vertices=vertices_label,
                                       triangles=triangles_label,
                                       name=("mesh_vertices_label_{}.pt".format(batch_index),
                                             "mesh_triangels_label_{}.pt".format(batch_index)))
            mcubes.export_obj(vertices=vertices_label.numpy(), triangles=triangles_label.numpy(),
                              filename=os.path.join(self.data_logger.path_plots,
                                                    "mesh_label_{}.obj".format(batch_index)))
        # Save logs
        self.data_logger.save()

    def extract_mash(self, occupancy_grid: torch.Tensor, cluster_offset: Optional[int] = 12 ** 3,
                     smooth: Optional[bool] = True, latent_vector: Optional[torch.Tensor] = None,
                     refine: Optional[bool] = False, simplify: Optional[bool] = True,
                     refinement_steps: Optional[int] = 200,
                     patches: Optional[torch.Tensor] = None,
                     large_patches: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method extracts a mesh from a given occupancy grid.
        :param occupancy_grid: (torch.Tensor) Occupancy grid
        :param cluster_offset: (Optional[int]) Offset of cluster size to be plotted smaller clusters will be ignored
        :param smooth: (Optional[bool]) If true binary occupancy grid gets smoothed before marching cubes algorithm
        :param latent_vector: (Optional[torch.Tensor]) Latent vector of model needed when refinement should be performed
        :param refine: (Optional[bool]) If true produced mesh will be refined by the gradients of the network
        :param simplify: (Optional[bool]) If true mesh gets simplified
        :param refinement_steps: (Optional[int]) Number of optimization steps to be utilized in refinement
        :param patches: (Optional[torch.Tensor]) Patches of the input volume
        """
        # To numpy
        occupancy_grid = occupancy_grid.cpu().numpy()
        # Cluster occupancy grid
        occupancy_grid = label(occupancy_grid)[0]
        # Get sizes of clusters
        cluster_indexes, cluster_sizes = np.unique(occupancy_grid, return_counts=True)
        # Iterate over clusters to eliminate clusters smaller than the offset
        for cluster_index, cluster_size in zip(cluster_indexes[1:], cluster_sizes[1:]):
            if (cluster_size < cluster_offset):
                occupancy_grid = np.where(occupancy_grid == cluster_index, 0, occupancy_grid)
        # Produce binary grid
        occupancy_grid = (occupancy_grid > 0.0).astype(float)
        # Remove batch dim if needed
        if occupancy_grid.ndim != 3:
            occupancy_grid = occupancy_grid.reshape(occupancy_grid.shape[-3:])
        # Apply distance transformation
        if smooth:
            occupancy_grid = mcubes.smooth(occupancy_grid)
        # Perform marching cubes
        vertices, triangles = mcubes.marching_cubes(occupancy_grid, 0. if smooth else 0.5)
        # Perform simplification if utilized
        if simplify:
            # Make open 3d mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            # Simplify mesh
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)
            # Get vertices and triangles
            vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        # Perform refinement if utilized
        if refine:
            vertices = torch.from_numpy(vertices)
            triangles = torch.from_numpy(triangles.astype(np.int64))
            # Init parameters
            vertices_parameter = nn.Parameter(vertices.clone().to(self.device), requires_grad=True)
            # Triangles to device
            triangles = triangles.long().to(self.device)
            # Init optimizer
            optimizer = torch.optim.RMSprop([vertices_parameter], lr=1e-03)
            # Optimization loop
            for _ in range(refinement_steps):
                # Rest gradients
                optimizer.zero_grad()
                # Get triangles vertices
                triangles_vertices = vertices_parameter[triangles]
                # Generate samples from dirichlet distribution
                samples = np.random.dirichlet((0.5, 0.5, 0.5), size=triangles_vertices.shape[0])
                samples = torch.from_numpy(samples).float().to(self.device)
                triangles_samples = (triangles_vertices.float() * samples.unsqueeze(dim=-1)).sum(dim=1)
                # Get different triangles vertices
                triangles_vertices_1 = triangles_vertices[:, 1] - triangles_vertices[:, 0]
                triangles_vertices_2 = triangles_vertices[:, 2] - triangles_vertices[:, 1]
                # Normalize triangles vertices
                triangles_vertices_normalized = torch.cross(triangles_vertices_1, triangles_vertices_2)
                triangles_vertices_normalized = triangles_vertices_normalized / \
                                                (triangles_vertices_normalized.norm(dim=1, keepdim=True) + 1e-10)
                if patches is not None:
                    # Get input patches
                    input_patches = patches[:, :,
                                    triangles_samples[:, 0].long(),
                                    triangles_samples[:, 1].long(),
                                    triangles_samples[:, 2].long()].transpose(1, 2).float()
                    # Get large patches if utilized
                    if large_patches is not None:
                        large_input_patches = large_patches[:, :,
                                              triangles_samples[:, 0].long(),
                                              triangles_samples[:, 1].long(),
                                              triangles_samples[:, 2].long()].transpose(1, 2).float()
                        # Downscale patches
                        large_input_patches = F.max_pool3d(large_input_patches[0], kernel_size=(2, 2, 2),
                                                           stride=(2, 2, 2))[None]
                        # Concat small and large patches
                        input_patches = torch.cat([input_patches, large_input_patches], dim=2)
                    # Data to device
                    input_patches = input_patches.float().to(self.device)
                else:
                    input_patches = None
                # Predict occupancy values
                triangles_vertices_predictions, _ = self.model(None, triangles_samples.unsqueeze(dim=0).float(),
                                                               input_patches, inference=True,
                                                               latent_vectors=latent_vector)
                # Calc targets
                targets = \
                    - autograd.grad([triangles_vertices_predictions.sum()], [triangles_samples], create_graph=True)[0]
                # Normalize targets
                targets_normalized = targets / (targets.norm(dim=1, keepdim=True) + 1e-10)
                # Calc target loss
                targets_loss = ((triangles_vertices_predictions - 0.5) ** 2).mean()
                # Calc normalization loss
                normalization_loss = ((triangles_vertices_normalized - targets_normalized) ** 2).sum(dim=1).mean()
                # Calc final loss
                loss = targets_loss + 0.01 * normalization_loss
                # Calc gradients
                loss.backward()
                # Perform optimization
                optimizer.step()
            return vertices_parameter.data.detach().cpu(), triangles.detach().cpu()
        return torch.from_numpy(vertices), torch.from_numpy(triangles.astype(np.int64))
