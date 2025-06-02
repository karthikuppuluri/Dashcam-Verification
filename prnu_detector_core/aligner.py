import cv2
import numpy as np

def align_images(ref_img: np.ndarray, img_to_align: np.ndarray) -> np.ndarray:
    """
    Align img_to_align to ref_img using ECC motion model.
    :param ref_img: Reference grayscale image
    :param img_to_align: Image to align (grayscale)
    :return: Aligned image
    """
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-7)
    cc, warp_matrix = cv2.findTransformECC(ref_img, img_to_align, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
    aligned = cv2.warpAffine(img_to_align, warp_matrix, (ref_img.shape[1], ref_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned