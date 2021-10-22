from typing import Dict, Any

import torch.nn as nn
from pade_activation_unit.utils import PAU

from .encoder import ConvolutionalEncoder, ConvolutionalEncoderSkip
from .decoder import ResidualCBNDecoder

# Vanilla O-Net setting
vanilla_o_net_bra_ts: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 320,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": False,
    # Utilize mid range patches
    "large_patches": False,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((4, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 5),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((320 + 3, 256), (256, 256), (256, 256), (256, 256), (256, 256)),
    # Patch mapping in decoder
    "decoder_patch_mapping": False
}

# O-Net wide encoder
vanilla_o_net_wide_bra_ts: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 320,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": False,
    # Utilize mid range patches
    "large_patches": False,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((4, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 5),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((320 + 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": False
}

# OSS-Net setting A (local patches)
oss_net_A_bra_ts: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 320,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": False,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((4, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 5),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((320 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting B (local patches, mid range patches)
oss_net_B_bra_ts: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 320,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": True,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((4, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 5),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((320 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting C1 (local patches, mid range patches, encoder skip connections)
oss_net_C1_bra_ts: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 320,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": False,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoderSkip,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((4, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 5),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((320 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting C2 (local patches, mid range patches, encoder skip connections)
oss_net_C2_bra_ts: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 320,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": True,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoderSkip,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((4, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 5),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((320 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting D1 (local patches, mid range patches, FIF)
oss_net_D1_bra_ts: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 320,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": False,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((4, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 5),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": True,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((320 + 35 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting D2 (local patches, mid range patches, FIF)
oss_net_D2_bra_ts: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 320,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": True,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((4, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 5),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": True,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((320 + 35 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting E1 (local patches, mid range patches, aux loss)
oss_net_E1_bra_ts: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 320,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": False,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((4, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": True,
    # Set output shape
    "encoder_output_shape": (8, 8, 5),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((320 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting E2 (local patches, mid range patches, aux loss)
oss_net_E2_bra_ts: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 320,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": True,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((4, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": True,
    # Set output shape
    "encoder_output_shape": (8, 8, 5),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((320 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting full
oss_net_full_bra_ts: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 320,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": False,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoderSkip,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((4, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": True,
    # Set output shape
    "encoder_output_shape": (8, 8, 5),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((320 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting full2
oss_net_full2_bra_ts: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 320,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": True,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoderSkip,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((4, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": True,
    # Set output shape
    "encoder_output_shape": (8, 8, 5),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((320 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# Vanilla O-Net setting
vanilla_o_net_lits: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 512,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": False,
    # Utilize mid range patches
    "large_patches": False,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((1, 8), (8, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 8),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((512 + 3, 256), (256, 256), (256, 256), (256, 256), (256, 256)),
    # Patch mapping in decoder
    "decoder_patch_mapping": False
}

# O-Net wide encoder
vanilla_o_net_wide_lits: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 512,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": False,
    # Utilize mid range patches
    "large_patches": False,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((1, 8), (8, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 8),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((512 + 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": False
}

# OSS-Net setting A (local patches)
oss_net_A_lits: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 512,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": False,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((1, 8), (8, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 8),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((512 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting B (local patches, mid range patches)
oss_net_B_lits: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 512,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": True,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((1, 8), (8, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 8),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((512 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting C1 (local patches, mid range patches, encoder skip connections)
oss_net_C1_lits: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 512,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": False,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoderSkip,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((1, 8), (8, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 8),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((512 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting C2 (local patches, mid range patches, encoder skip connections)
oss_net_C2_lits: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 512,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": True,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoderSkip,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((1, 8), (8, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 8),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((512 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting D1 (local patches, mid range patches, FIF)
oss_net_D1_lits: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 512,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": False,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((1, 8), (8, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 8),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": True,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((512 + 35 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting D2 (local patches, mid range patches, FIF)
oss_net_D2_lits: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 512,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": True,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((1, 8), (8, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": False,
    # Set output shape
    "encoder_output_shape": (8, 8, 8),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": True,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((512 + 35 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting E1 (local patches, mid range patches, aux loss)
oss_net_E1_lits: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 512,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": False,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((1, 8), (8, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": True,
    # Set output shape
    "encoder_output_shape": (8, 8, 8),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((512 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting E2 (local patches, mid range patches, aux loss)
oss_net_E2_lits: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 512,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": True,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoder,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((1, 8), (8, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": True,
    # Set output shape
    "encoder_output_shape": (8, 8, 8),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((512 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting full
oss_net_full_lits: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 512,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": False,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoderSkip,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((1, 8), (8, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": True,
    # Set output shape
    "encoder_output_shape": (8, 8, 8),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((512 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}

# OSS-Net setting full2
oss_net_full2_lits: Dict[str, Any] = {
    # Type of activation function to be utilized in the whole network
    "activation": PAU,
    # Number of output classes to be predicted
    "number_of_classes": 1,
    # Latent vector size
    "latent_features": 512,
    # Dropout factor to be utilized in the whole network
    "dropout": 0.0,
    # Utilize patches
    "patches": True,
    # Utilize mid range patches
    "large_patches": True,

    # Type of encoder to be used
    "encoder": ConvolutionalEncoderSkip,
    # Type of normalization utilized in encoder
    "encoder_normalization": nn.BatchNorm3d,
    # Channels to be used in encoder
    "encoder_channels": ((1, 8), (8, 16), (16, 32), (32, 64), (64, 128)),
    # Utilize encoder segmentation mapping for auxiliary loss
    "encoder_segmentation_mapping": True,
    # Set output shape
    "encoder_output_shape": (8, 8, 8),

    # Utilized fourier input features in decoder
    "decoder_fourier_input_features": False,
    # Type of decoder to be utilized
    "decoder": ResidualCBNDecoder,
    # Decoder features to be used
    "decoder_features": ((512 + 3 + 7 ** 3, 512), (512, 512), (512, 512), (512, 512), (512, 512)),
    # Patch mapping in decoder
    "decoder_patch_mapping": True
}
