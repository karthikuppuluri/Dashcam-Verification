import numpy as np
import os
import cv2
from .prnu_utils import extract_noise_residual, flatten_and_center, normalized_cross_correlation
from .loader import extract_frames
from .aligner import align_images

def compute_camera_fingerprint(frame_paths: list) -> np.ndarray:
    residuals = []
    for path in frame_paths:
        img = cv2.imread(path)
        noise = extract_noise_residual(img)
        residuals.append(noise)
    average_residual = np.mean(residuals, axis=0)
    return average_residual

def detect_forgery(video_path: str, frame_step: int = 20, threshold: float = 0.01) -> dict:
    temp_dir = "temp_frames"
    extract_frames(video_path, temp_dir, step=frame_step)
    frame_files = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".png")])

    # Create camera fingerprint from the first few frames
    fingerprint = compute_camera_fingerprint(frame_files[:10])

    results = {}
    for path in frame_files:
        img = cv2.imread(path)
        noise = extract_noise_residual(img)

        fingerprint_vec = flatten_and_center(fingerprint)
        noise_vec = flatten_and_center(noise)

        corr = normalized_cross_correlation(fingerprint_vec, noise_vec)
        results[path] = corr

    # Clean up
    for f in frame_files:
        os.remove(f)
    os.rmdir(temp_dir)

    return results