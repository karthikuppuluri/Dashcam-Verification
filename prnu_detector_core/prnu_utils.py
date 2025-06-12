import numpy as np
import cv2
from bm3d import bm3d_rgb, bm3d
from multiprocessing import Pool, cpu_count
from functools import partial

def coefficient_of_variation(block):
    """
    Calculate coefficient of variation for a block.
    :param block: Input image block as numpy array
    :return: Coefficient of variation (CV)
    """
    # Optimize CV calculation for speed
    block_flat = block.ravel()
    mean = np.mean(block_flat)
    if mean == 0:
        return 0
    std = np.std(block_flat)
    return std / mean

def get_adjusted_sigma(cv, base_sigma):
    """
    Get adjusted sigma based on coefficient of variation.
    :param cv: Coefficient of variation
    :param base_sigma: Base sigma value
    :return: Adjusted sigma value
    """
    if cv < 0.05:  # Flat region
        return base_sigma * 0.7
    elif cv < 0.1:  # Moderate texture
        return base_sigma * 0.85
    return base_sigma  # Complex texture

def process_block(block_data):
    """
    Process a single block with appropriate BM3D parameters.
    :param block_data: Tuple of (block, sigma)
    :return: Tuple of (denoised_block, block_index)
    """
    block, sigma, block_index = block_data
    
    # Skip processing if block is too small
    if block.shape[0] < 8 or block.shape[1] < 8:
        return block, block_index
        
    cv = coefficient_of_variation(block)
    adjusted_sigma = get_adjusted_sigma(cv, sigma)
    
    # Convert block to float32 and normalize
    block_norm = block.astype(np.float32) / 255.0
    
    # Process based on block type
    if len(block.shape) == 3:
        denoised = np.zeros_like(block_norm)
        # Process all channels at once for better efficiency
        for i in range(3):
            denoised[:,:,i] = bm3d(block_norm[:,:,i], sigma_psd=adjusted_sigma/255.0)
    else:
        denoised = bm3d(block_norm, sigma_psd=adjusted_sigma/255.0)
    
    return denoised, block_index

def extract_noise_residual(image: np.ndarray, sigma=5, block_size=128) -> np.ndarray:
    """
    Extract the noise residual from an image using parallel adaptive BM3D denoising.
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
    
    # Prepare blocks for parallel processing
    blocks = []
    positions = []
    
    # Split image into blocks
    block_index = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Get block boundaries
            i_end = min(i + block_size, h)
            j_end = min(j + block_size, w)
            
            # Extract block
            block = image[i:i_end, j:j_end].copy()  # Make a copy to ensure it's contiguous
            blocks.append((block, sigma, block_index))
            positions.append((i, i_end, j, j_end))
            block_index += 1
    
    # Process blocks in parallel using multiprocessing
    n_workers = min(cpu_count(), 8)  # Use at most 8 cores
    with Pool(processes=n_workers) as pool:
        results = pool.map(process_block, blocks)
    
    # Reconstruct the image
    for (denoised_block, block_idx), (i_start, i_end, j_start, j_end) in zip(results, positions):
        denoised[i_start:i_end, j_start:j_end] = denoised_block
    
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