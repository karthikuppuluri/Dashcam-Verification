# File: preprocess_grayscale.py

import os
import cv2
from prnu_detector_core import to_grayscale  # This imports the function from prnu_utils.py

frame_output_dir = "frames/"  # Where your extracted frames are stored
grayscale_dir = "grayscale_frames/"
os.makedirs(grayscale_dir, exist_ok=True)

# Iterate through each extracted frame
for frame_file in sorted(os.listdir(frame_output_dir)):
    frame_path = os.path.join(frame_output_dir, frame_file)
    image = cv2.imread(frame_path)                 # Read the color image
    gray = to_grayscale(image)                     # Convert to grayscale
    save_path = os.path.join(grayscale_dir, frame_file)
    cv2.imwrite(save_path, gray)                   # Save grayscale image