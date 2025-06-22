import numpy as np

import eclipsetools.utils.circle_finder
from eclipsetools.preprocessing import filtering
from eclipsetools.preprocessing import masking


def preprocess_for_alignment(rgb_image):
    image = np.mean(rgb_image, axis=2, dtype=np.float32)
    moon = eclipsetools.utils.circle_finder.find_circle(image, min_radius=400, max_radius=700)
    assert moon is not None
    saturated_radius = masking.estimate_saturated_radius(moon, rgb_image)
    moon_mask_radius = 1.1 * (saturated_radius if saturated_radius else moon.radius)
    return _mask_and_filter(image, moon.center, moon_mask_radius)


def _mask_and_filter(image: np.ndarray, moon_center: tuple, moon_mask_radius: float) -> np.ndarray:
    window_mask = masking.hann_window_mask(image.shape)
    moon_mask = masking.annulus_mask(image.shape, moon_center, moon_mask_radius)
    mask = window_mask * moon_mask
    # TODO: Parametrize sigma, which is used to control the amount of rotational blur used in the tangential high-pass filter
    filtered_image = image - filtering.rotational_blur(image,
                                                       sigma=0.5,
                                                       center=moon_center)
    image_for_alignment = mask * filtered_image
    return image_for_alignment
