# This script generates volumes including the border regions of given MRI labels (LITS)
import os

import nibabel
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import sobel, generic_gradient_magnitude
from tqdm import tqdm

paths = ["LITS Challenge/TrainingData", "LITS Challenge/TestData"]

for path in paths:
    # Init progress bar
    progress_bar = tqdm(total=len(os.listdir(path=os.path.join(path, "segmentation"))))
    for index, file in enumerate(os.listdir(path=os.path.join(path, "segmentation"))):
        progress_bar.update(n=1)
        # Load segmentation
        segmentation = nibabel.load(filename=os.path.join(path, "segmentation", file)).get_fdata() > 0
        # Calc gradients
        gradients = generic_gradient_magnitude(segmentation, sobel).astype(np.float32)
        # Erase gradient to get border region
        with torch.no_grad():
            border_region = F.max_pool3d(torch.from_numpy(gradients)[None, None].cuda(),
                                         kernel_size=(13, 13, 13), stride=(1, 1, 1), padding=(6, 6, 6))[0, 0].cpu()
        # Save borders
        torch.save(border_region.bool(),
                   os.path.join(path, "border", file.replace("segmentation", "border").replace(".nii", ".pt")))
    # Close progress bar
    progress_bar.close()
