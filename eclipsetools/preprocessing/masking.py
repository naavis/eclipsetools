import cv2
import numpy as np

from eclipsetools.utils.circle_finder import DetectedCircle


def hann_window_mask(shape: tuple) -> np.ndarray:
    assert len(shape) == 2, "Shape must be a 2D tuple (height, width)."
    return cv2.createHanningWindow(shape[::-1], cv2.CV_32F)


def annulus_mask(shape: np.ndarray,
                 mask_center: tuple,
                 inner_radius: float,
                 outer_radius: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)

    cv2.circle(
        img=mask,
        center=(int(mask_center[1]), int(mask_center[0])),
        radius=int(outer_radius),
        color=(1.0,),
        thickness=-1)
    cv2.circle(
        img=mask,
        center=(int(mask_center[1]), int(mask_center[0])),
        radius=int(inner_radius),
        color=(0.0,),
        thickness=-1)
    cv2.GaussianBlur(
        src=mask,
        ksize=(0, 0),
        sigmaX=20,
        sigmaY=20,
        dst=mask)

    return mask


def estimate_saturated_radius(moon_params: DetectedCircle,
                              raw_image: np.ndarray) -> float | None:
    saturated_pixels = cv2.erode(np.max(raw_image, axis=2), kernel=np.ones((3, 3), np.float32), iterations=1) > 0.999
    saturated_radius = None
    if np.any(saturated_pixels):
        distances = np.linalg.norm(np.argwhere(saturated_pixels) - moon_params.center, axis=1)
        saturated_radius = np.max(distances)
    return saturated_radius
