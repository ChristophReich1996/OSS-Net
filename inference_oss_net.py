from argparse import ArgumentParser
import os

import setproctitle

# Manage command line arguments
parser = ArgumentParser()

parser.add_argument("--cuda_devices", default="0", type=str,
                    help="String of cuda device indexes to be used. Indexes must be separated by a comma.")
parser.add_argument("--cpu", default=False, action="store_true",
                    help="Binary flag. If set all operations are performed on the CPU.")
parser.add_argument("--load_model", default="", type=str,
                    help="Path to model to be loaded.")
parser.add_argument("--dataset", default="BraTS", type=str, choices=["BraTS", "LITS"],
                    help="Dataset to be utilized")
parser.add_argument("--dataset_path", default="/home/creich/BraTS2020", type=str,
                    help="Path to dataset")
parser.add_argument("--network_config", default="", type=str,
                    choices=["vanilla_o_net", "vanilla_o_net_wide", "A", "B", "C1", "C2", "D1", "D2", "E1", "E2",
                             "full", "full2"],
                    help="Type of network configuration to be utilized.")

# Get arguments
args = parser.parse_args()

# Set device type
device = "cpu" if args.cpu else "cuda"

# Set cuda devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

# Set process title
setproctitle.setproctitle("OSS-Net inference")

import torch

import oss_net
import data
import misc
from oss_net.model_wrapper import ModelWrapper

if __name__ == '__main__':
    # Init data logger
    data_logger = misc.Logger(experiment_path_extension="_inference_" + args.network_config + "_" + args.dataset)
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
            print("OSS-Net full utilized.")
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
            print("OSS-Net full utilized.")
            config = oss_net.oss_net_full2_lits
        else:
            print("OSS-Net E2 utilized.")
            config = oss_net.oss_net_E2_lits
    model = oss_net.OSSNet(config=config).to(device)
    data_logger.log_hyperparameter(hyperparameter_dict=config)
    if args.load_model != "":
        print("Load model from: {}".format(args.load_model))
        model.load_state_dict(
            torch.load(args.load_model))
    # Init inference dataset
    if args.dataset == "BraTS":
        print("BraTS 2020 dataset utilized")
        inference_dataset = data.BraTS2020SegmentationInference(
            os.path.join(args.dataset_path, "MICCAI_BraTS2020_TestData"), normalize_coordinates=False,
            patches=config["patches"], large_patches=config["large_patches"])
    else:
        print("LITS dataset utilized")
        inference_dataset = data.BraTS2020SegmentationInference(
            os.path.join(args.dataset_path, "TestData"), normalize_coordinates=False,
            patches=config["patches"], large_patches=config["large_patches"])
    # Init model wrapper
    model_wrapper = ModelWrapper(model=model,
                                 optimizer=None,
                                 loss_function=None,
                                 loss_function_latent_segmentation=None,
                                 training_dataset=None,
                                 test_dataset=None,
                                 learning_rate_schedule=None,
                                 device=device,
                                 segmentation_loss_factor=None,
                                 data_logger=data_logger)
    # Perform inference
    model_wrapper.inference(inference_dataset=inference_dataset)
