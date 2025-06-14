import cv2
import numpy as np
from scipy.signal.windows import hann


def hann_window_mask(shape: tuple) -> np.ndarray:
    return np.outer(hann(shape[0]), hann(shape[1]))

def circle_mask(image: np.ndarray,
                mask_center: tuple,
                mask_radius: float) -> np.ndarray:
    mask = np.zeros_like(image)
    mask = 1 - cv2.circle(
        img=mask,
        center=(int(mask_center[1]), int(mask_center[0])),
        radius=int(mask_radius),
        color=(1.0, 1.0, 1.0, 1.0),
        thickness=-1)
    mask = cv2.GaussianBlur(
        src=mask,
        ksize=(0, 0),
        sigmaX=mask.shape[0] / 100,
        sigmaY=mask.shape[0] / 100)
    return mask

