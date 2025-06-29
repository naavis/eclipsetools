import numpy as np

import eclipsetools.utils.circle_finder
from eclipsetools.preprocessing import filtering
from eclipsetools.preprocessing import masking


def preprocess_for_alignment(rgb_image):
    image = np.mean(rgb_image, axis=2, dtype=np.float32)
    moon = eclipsetools.utils.circle_finder.find_circle(image, min_radius=400, max_radius=600)
    assert moon is not None
    saturated_radius = masking.estimate_saturated_radius(moon, rgb_image)
    moon_mask_radius = 1.2 * (saturated_radius if saturated_radius else moon.radius)
    preproc = _mask_and_filter(image, moon.center, moon_mask_radius)
    return preproc


def _mask_and_filter(image: np.ndarray, moon_center: tuple, moon_mask_radius: float) -> np.ndarray:
    # TODO: Parametrize sigma, which is used to control the amount of rotational blur used in the tangential high-pass filter
    blurred_image = filtering.rotational_blur(image,
                                              sigma=2.0,
                                              center=moon_center)
    filtered_image = image - blurred_image

    annulus_mask = masking.annulus_mask(image.shape, moon_center, moon_mask_radius)
    image_for_alignment = annulus_mask * filtered_image
    return image_for_alignment
