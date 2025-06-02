import numpy as np
import cv2
from bm3d import bm3d_rgb

def extract_noise_residual(image: np.ndarray, sigma=5) -> np.ndarray:
    """
    Extract the noise residual from an RGB image using BM3D denoising.
    :param image: Input image as a NumPy array (H x W x 3).
    :param sigma: Estimated noise standard deviation.
    :return: Noise residual (same shape as input image).
    """
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    denoised = bm3d_rgb(image, sigma_psd=sigma / 255.0)
    residual = image - denoised
    return residual

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.
    :param image: RGB image as a NumPy array (H x W x 3)
    :return: Grayscale image (H x W)
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def flatten_and_center(patch: np.ndarray) -> np.ndarray:
    """
    Flatten a 2D patch and subtract the mean.
    :param patch: 2D array
    :return: Zero-mean 1D array
    """
    vec = patch.flatten()
    return vec - np.mean(vec)

def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute normalized cross-correlation between two zero-mean 1D vectors.
    :param a: 1D numpy array
    :param b: 1D numpy array
    :return: Correlation score
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)