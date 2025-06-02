"""
PRNU Detector Core Package

This package provides core functionalities for detecting image and video tampering
based on Photo Response Non-Uniformity (PRNU) noise patterns.

Exposed Modules:
- prnu_utils: Noise extraction, normalization, correlation tools.
- loader: Frame extraction from video files.
- aligner: Image alignment using ECC motion model.
"""

from .prnu_utils import (
    extract_noise_residual,
    to_grayscale,
    flatten_and_center,
    normalized_cross_correlation
)

from .loader import extract_frames
from .aligner import align_images

# Defines the public API for the package
__all__ = [
    "extract_noise_residual",
    "to_grayscale",
    "flatten_and_center",
    "normalized_cross_correlation",
    "extract_frames",
    "align_images"
]