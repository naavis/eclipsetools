import cv2
import numpy as np

from eclipsetools.utils.circle_finder import DetectedCircle
from eclipsetools.utils.memorycache import memory


@memory.cache
def hann_window_mask(shape: tuple) -> np.ndarray:
    return cv2.createHanningWindow(shape[::-1], cv2.CV_32F)


@memory.cache
def annulus_mask(shape: np.ndarray,
                 mask_center: tuple,
                 mask_radius: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)

    cv2.circle(img=mask,
               center=(int(mask_center[1]), int(mask_center[0])),
               radius=int(mask_radius * 2.0),
               color=(1.0,),
               thickness=-1)
    cv2.circle(img=mask,
               center=(int(mask_center[1]), int(mask_center[0])),
               radius=int(mask_radius),
               color=(0.0,),
               thickness=-1)
    mask = cv2.GaussianBlur(
        src=mask,
        ksize=(0, 0),
        sigmaX=10,
        sigmaY=10)
    return mask


def estimate_saturated_radius(moon_params: DetectedCircle,
                              raw_image: np.ndarray) -> float | None:
    saturated_pixels = cv2.erode(np.max(raw_image, axis=2), kernel=np.ones((3, 3), np.float32), iterations=1) > 0.999
    saturated_radius = None
    if np.any(saturated_pixels):
        distances = np.linalg.norm(np.argwhere(saturated_pixels) - moon_params.center, axis=1)
        saturated_radius = np.max(distances)
    return saturated_radius
