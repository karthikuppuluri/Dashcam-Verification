import numpy as np
import cv2
from bm3d import bm3d_rgb, bm3d

def coefficient_of_variation(block):
    """
    Calculate coefficient of variation for a block.
    :param block: Input image block as numpy array
    :return: Coefficient of variation (CV)
    """
    mean = np.mean(block)
    std = np.std(block)
    return std / mean if mean != 0 else 0

def get_block_size(cv):
    """
    Determine block size based on coefficient of variation.
    :param cv: Coefficient of variation
    :return: Appropriate block size
    """
    if cv < 0.05:
        return 8
    elif cv < 0.1:
        return 10
    else:
        return 12

def process_block(block, sigma):
    """
    Process a single block with appropriate BM3D parameters.
    :param block: Input image block
    :param sigma: Noise standard deviation
    :return: Denoised block
    """
    cv = coefficient_of_variation(block)
    
    # Adjust sigma based on CV - use smaller sigma for flat regions
    if cv < 0.05:  # Flat region
        adjusted_sigma = sigma * 0.7
    elif cv < 0.1:  # Moderate texture
        adjusted_sigma = sigma * 0.85
    else:  # Complex texture
        adjusted_sigma = sigma
    
    # Convert block to float32 and normalize
    block_norm = block.astype(np.float32) / 255.0
    
    # Apply BM3D with adjusted parameters
    if len(block.shape) == 3:  # RGB block
        # For RGB, process each channel separately
        denoised = np.zeros_like(block_norm)
        for i in range(3):  # Process each color channel
            denoised[:,:,i] = bm3d(block_norm[:,:,i], sigma_psd=adjusted_sigma/255.0)
    else:  # Grayscale block
        denoised = bm3d(block_norm, sigma_psd=adjusted_sigma/255.0)
    
    return denoised

def extract_noise_residual(image: np.ndarray, sigma=5, block_size=64) -> np.ndarray:
    """
    Extract the noise residual from an image using adaptive block-size BM3D denoising.
    :param image: Input image as a NumPy array (H x W x 3 for RGB or H x W for grayscale).
    :param sigma: Estimated noise standard deviation.
    :param block_size: Size of blocks to process independently.
    :return: Noise residual (same shape as input image).
    """
    # Convert to float32 and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Get image dimensions
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = 1
        image = image.reshape(h, w, 1)
    
    # Initialize output array
    denoised = np.zeros_like(image)
    
    # Process image in blocks
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Get current block
            block = image[i:min(i+block_size, h), j:min(j+block_size, w)]
            
            # Process block
            denoised_block = process_block(block, sigma)
            
            # Store result
            denoised[i:min(i+block_size, h), j:min(j+block_size, w)] = denoised_block
    
    # Calculate residual
    residual = image - denoised
    
    # Reshape back if input was grayscale
    if c == 1:
        residual = residual.reshape(h, w)
    
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