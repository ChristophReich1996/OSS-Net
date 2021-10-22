from argparse import ArgumentParser
import os

# Manage command line arguments
parser = ArgumentParser()

parser.add_argument("--train", default=False, action="store_true",
                    help="Binary flag. If set training will be performed.")
parser.add_argument("--test", default=False, action="store_true",
                    help="Binary flag. If set testing will be performed.")
parser.add_argument("--cuda_devices", default="0, 1", type=str,
                    help="String of cuda device indexes to be used. Indexes must be separated by a comma.")
parser.add_argument("--cpu", default=False, action="store_true",
                    help="Binary flag. If set all operations are performed on the CPU.")
parser.add_argument("--epochs", default=50, type=int,
                    help="Number of epochs to perform while training.")
parser.add_argument("--batch_size", default=8, type=int,
                    help="Number of epochs to perform while training.")
parser.add_argument("--training_samples", default=2 ** 14, type=int,
                    help="Number of coordinates to be samples during training.")
parser.add_argument("--load_model", default="", type=str,
                    help="Path to model to be loaded.")
parser.add_argument("--segmentation_loss_factor", default=0.1, type=float,
                    help="Auxiliary segmentation loss factor to be utilized.")
parser.add_argument("--network_config", default="", type=str,
                    choices=["vanilla_o_net", "vanilla_o_net_wide", "A", "B", "C1", "C2", "D1", "D2", "E1", "E2",
                             "full", "full2"],
                    help="Type of network configuration to be utilized.")
parser.add_argument("--dataset", default="BraTS", type=str, choices=["BraTS", "LITS"],
                    help="Dataset to be utilized.")
parser.add_argument("--dataset_path", default="BraTS2020", type=str,
                    help="Path to dataset.")
parser.add_argument("--uniform_sampling", default=False, action="store_true",
                    help="Binary flag. If set locations are sampled uniformly during training.")

# Get arguments
args = parser.parse_args()

# Avoid data loader bug
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2 ** 12, rlimit[1]))

# Set device type
device = "cpu" if args.cpu else "cuda"

# Set cuda devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch_optimizer

import oss_net
import data
import misc
import loss
from oss_net import ModelWrapper

if __name__ == '__main__':
    # Init data logger
    experiment_path_extension = "_training_" + args.network_config + "_" + args.dataset
    if args.uniform_sampling:
        experiment_path_extension += "_uniform_sampling_"
    else:
        experiment_path_extension += "_weighted_sampling_"
    experiment_path_extension += "_{}".format(args.training_samples)

    data_logger = misc.Logger(experiment_path_extension=experiment_path_extension)
    # Init model
    if args.dataset == "BraTS":
        if args.network_config == "vanilla_o_net":
            print("Vanilla O-Net utilized.")
            config = oss_net.vanilla_o_net_bra_ts
        elif args.network_config == "vanilla_o_net_wide":
            print("Vanilla O-Net wide utilized.")
            config = oss_net.vanilla_o_net_wide_bra_ts
        elif args.network_config == "A":
            print("OSS-Net A utilized.")
            config = oss_net.oss_net_A_bra_ts
        elif args.network_config == "B":
            print("OSS-Net B utilized.")
            config = oss_net.oss_net_B_bra_ts
        elif args.network_config == "C1":
            print("OSS-Net C1 utilized.")
            config = oss_net.oss_net_C1_bra_ts
        elif args.network_config == "C2":
            print("OSS-Net C2 utilized.")
            config = oss_net.oss_net_C2_bra_ts
        elif args.network_config == "D1":
            print("OSS-Net D1 utilized.")
            config = oss_net.oss_net_D1_bra_ts
        elif args.network_config == "D2":
            print("OSS-Net D2 utilized.")
            config = oss_net.oss_net_D2_bra_ts
        elif args.network_config == "E1":
            print("OSS-Net E1 utilized.")
            config = oss_net.oss_net_E1_bra_ts
        elif args.network_config == "full":
            print("OSS-Net full utilized.")
            config = oss_net.oss_net_full_bra_ts
        elif args.network_config == "full2":
            print("OSS-Net full 2 utilized.")
            config = oss_net.oss_net_full2_bra_ts
        else:
            print("OSS-Net E2 utilized.")
            config = oss_net.oss_net_E2_bra_ts
    else:
        if args.network_config == "vanilla_o_net":
            print("Vanilla O-Net utilized.")
            config = oss_net.vanilla_o_net_lits
        elif args.network_config == "vanilla_o_net_wide":
            print("Vanilla O-Net wide utilized.")
            config = oss_net.vanilla_o_net_wide_lits
        elif args.network_config == "A":
            print("OSS-Net A utilized.")
            config = oss_net.oss_net_A_lits
        elif args.network_config == "B":
            print("OSS-Net B utilized.")
            config = oss_net.oss_net_B_lits
        elif args.network_config == "C1":
            print("OSS-Net C1 utilized.")
            config = oss_net.oss_net_C1_lits
        elif args.network_config == "C2":
            print("OSS-Net C2 utilized.")
            config = oss_net.oss_net_C2_lits
        elif args.network_config == "D1":
            print("OSS-Net D1 utilized.")
            config = oss_net.oss_net_D1_lits
        elif args.network_config == "D2":
            print("OSS-Net D2 utilized.")
            config = oss_net.oss_net_D2_lits
        elif args.network_config == "E1":
            print("OSS-Net E1 utilized.")
            config = oss_net.oss_net_E1_lits
        elif args.network_config == "full":
            print("OSS-Net full utilized.")
            config = oss_net.oss_net_full_lits
        elif args.network_config == "full2":
            print("OSS-Net full 2 utilized.")
            config = oss_net.oss_net_full2_lits
        else:
            print("OSS-Net E2 utilized.")
            config = oss_net.oss_net_E2_lits
    model = oss_net.OSSNet(config=config)
    data_logger.log_hyperparameter(hyperparameter_dict=config)
    # Load existing model if utilized
    if args.load_model != "":
        print("Load model from: {}".format(args.load_model))
        model.load_state_dict(
            torch.load(args.load_mode))
    data_logger.save()
    # Model to device
    model.to(device)
    # Show number of parameters
    print("# model parameters:", sum([parameter.numel() for parameter in model.parameters()]))
    # Init optimizer
    optimizer = torch_optimizer.Lookahead(torch_optimizer.RAdam(model.parameter_dicts(lr=3e-04)), k=5, alpha=0.8)
    # Init data parallel
    model = nn.DataParallel(module=model).to(device)
    # Init training dataset
    if args.dataset == "BraTS":
        print("BraTS 2020 dataset utilized")
        training_dataset = DataLoader(data.BraTS2020SegmentationTraining(
            os.path.join(args.dataset_path, "MICCAI_BraTS2020_TrainingData"), normalize_coordinates=False,
            patches=config["patches"], large_patches=config["large_patches"],
            border_balance=0.0 if args.uniform_sampling else 0.5, samples=args.training_samples),
            batch_size=args.batch_size, shuffle=True, num_workers=12,
            collate_fn=data.collate_function, drop_last=True, pin_memory=True)
        # Init test dataset
        test_dataset = DataLoader(data.BraTS2020SegmentationTest(
            os.path.join(args.dataset_path, "MICCAI_BraTS2020_TestData"), normalize_coordinates=False,
            patches=config["patches"], large_patches=config["large_patches"]),
            batch_size=len(model.device_ids), shuffle=False, num_workers=len(model.device_ids),
            collate_fn=data.collate_function, pin_memory=True)
    else:
        print("LITS dataset utilized")
        training_dataset = DataLoader(data.LITSSegmentationTraining(
            os.path.join(args.dataset_path, "TrainingData"), normalize_coordinates=False,
            patches=config["patches"], large_patches=config["large_patches"],
            border_balance=0.0 if args.uniform_sampling else 0.5, samples=args.training_samples),
            batch_size=args.batch_size, shuffle=True, num_workers=12,
            collate_fn=data.collate_function, drop_last=True, pin_memory=True)
        # Init test dataset
        test_dataset = DataLoader(data.LITSSegmentationTest(
            os.path.join(args.dataset_path, "TestData"), normalize_coordinates=False, patches=config["patches"],
            large_patches=config["large_patches"]),
            batch_size=len(model.device_ids), shuffle=False, num_workers=len(model.device_ids),
            collate_fn=data.collate_function, pin_memory=True)
    # Init learning rate schedule
    learning_rate_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[20, 30], gamma=0.1)
    # Init model wrapper
    model_wrapper = ModelWrapper(model=model,
                                 optimizer=optimizer,
                                 loss_function=loss.BinaryCrossEntropyLoss(bootstrap=False,
                                                                           label_smoothing=0.),
                                 loss_function_latent_segmentation=loss.BinaryCrossEntropyLoss(bootstrap=False,
                                                                                               label_smoothing=0.),
                                 training_dataset=training_dataset,
                                 test_dataset=test_dataset,
                                 learning_rate_schedule=learning_rate_schedule,
                                 device=device,
                                 segmentation_loss_factor=args.segmentation_loss_factor,
                                 data_logger=data_logger)
    # Perform training
    if args.train:
        model_wrapper.train(epochs=args.epochs, validate_after_n_epochs=5 if args.test else 1e+10,
                            save_model_after_n_epochs=5)
