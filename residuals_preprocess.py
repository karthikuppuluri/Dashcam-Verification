from prnu_detector_core import extract_noise_residual
import os
import cv2
import numpy as np


residual_dir = "data/residuals"
os.makedirs(residual_dir, exist_ok=True)

for f in sorted(os.listdir("frames")):  # must use color frames
    path = os.path.join("frames", f)
    img = cv2.imread(path)  # RGB frame
    residual = extract_noise_residual(img)
    residual = (residual * 255).astype(np.uint8)  # scale to view or save
    cv2.imwrite(os.path.join(residual_dir, f), residual)