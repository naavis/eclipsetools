import cv2
import numpy as np

import eclipsetools.preprocessing.filtering
import eclipsetools.preprocessing.masking
import eclipsetools.utils.circle_finder


def preprocess_for_alignment(rgb_image):
    image = np.mean(rgb_image, axis=2)
    moon = eclipsetools.utils.circle_finder.find_circle(image, min_radius=400, max_radius=700)
    assert moon is not None
    saturated_radius = eclipsetools.preprocessing.masking.estimate_saturated_radius(moon, rgb_image)
    moon_mask_radius = 1.1 * (saturated_radius if saturated_radius else moon.radius)
    return mask_and_filter(image, moon.center, moon_mask_radius)


def mask_and_filter(image: np.ndarray, moon_center: tuple, moon_mask_radius: float) -> np.ndarray:
    window_mask = eclipsetools.preprocessing.masking.hann_window_mask(image.shape)
    moon_mask = eclipsetools.preprocessing.masking.circle_mask(image.shape, moon_center, moon_mask_radius)
    mask = window_mask * moon_mask
    filtered_image = image - eclipsetools.preprocessing.filtering.rotational_blur(image, max_angle=2,
                                                                                  center=moon_center)
    image_for_alignment = mask * filtered_image
    return image_for_alignment
