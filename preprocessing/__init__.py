import numpy as np
import scipy

import preprocessing.masking
import preprocessing.filtering
import utils.circlefinder


def preprocess_for_alignment(rgb_image):
    image = np.mean(rgb_image, axis=2)
    moon = utils.circlefinder.find_circle(image, min_radius=400, max_radius=700)
    assert moon is not None
    saturated_radius = estimate_saturated_radius(moon, rgb_image)
    moon_mask_radius = 1.2 * (saturated_radius if saturated_radius else moon.radius)
    return mask_and_filter(image, moon.center, moon_mask_radius)


def estimate_saturated_radius(moon_params: utils.circlefinder.DetectedCircle, raw_image: np.ndarray) -> float | None:
    saturated_pixels = scipy.ndimage.median_filter(np.max(raw_image, axis=2), size=3) > 0.999
    saturated_radius = None
    if np.any(saturated_pixels):
        distances = np.linalg.norm(np.argwhere(saturated_pixels) - moon_params.center, axis=1)
        saturated_radius = np.max(distances)
    return saturated_radius


def mask_and_filter(image: np.ndarray, moon_center: tuple, moon_mask_radius: float) -> np.ndarray:
    window_mask = preprocessing.masking.hann_window_mask(image.shape)
    moon_mask = preprocessing.masking.circle_mask(image, moon_center, moon_mask_radius)
    mask = window_mask * moon_mask
    filtered_image = image - preprocessing.filtering.rotational_blur(image, max_angle=2, center=moon_center)
    image_for_alignment = mask * filtered_image
    return image_for_alignment
