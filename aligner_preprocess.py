import numpy as np
from prnu_detector_core import align_images
import os
import cv2

aligned_dir = "data/aligned_frames"
os.makedirs(aligned_dir, exist_ok=True)
grayscale_dir = "grayscale_frames/"


frame_files = sorted(os.listdir(grayscale_dir))
ref = cv2.imread(os.path.join(grayscale_dir, frame_files[0]), cv2.IMREAD_GRAYSCALE)

for i, f in enumerate(frame_files):
    img = cv2.imread(os.path.join(grayscale_dir, f), cv2.IMREAD_GRAYSCALE)
    aligned = align_images(ref, img)
    cv2.imwrite(os.path.join(aligned_dir, f), aligned)