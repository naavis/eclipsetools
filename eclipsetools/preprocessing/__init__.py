import numpy as np

import eclipsetools.utils.circle_finder
from eclipsetools.preprocessing import filtering
from eclipsetools.preprocessing import masking
from eclipsetools.preprocessing.masking import circle_mask, find_mask_radii_px


def preprocess_for_alignment(rgb_image: np.ndarray,
                             mask_inner_radius_multiplier: float,
                             mask_outer_radius_multiplier: float) -> np.ndarray:
    image = np.mean(rgb_image, axis=2, dtype=np.float32)
    moon = eclipsetools.utils.circle_finder.find_circle(image, min_radius=400, max_radius=600)
    assert moon is not None, "Moon not found in the image. Please check the input image."
    mask_inner_radius_px, mask_outer_radius_px = find_mask_radii_px(rgb_image,
                                                                    moon,
                                                                    mask_inner_radius_multiplier,
                                                                    mask_outer_radius_multiplier)

    preproc = _mask_and_filter(image, moon.center, mask_inner_radius_px, mask_outer_radius_px)
    return preproc


def _mask_and_filter(image: np.ndarray,
                     moon_center: tuple,
                     mask_inner_radius_px: float,
                     mask_outer_radius_px: float | None) -> np.ndarray:
    # TODO: Parametrize sigma, which is used to control the amount of rotational blur used in the tangential high-pass filter
    blurred_image = filtering.rotational_blur(image,
                                              sigma=2.0,
                                              center=moon_center)
    filtered_image = image - blurred_image

    if mask_outer_radius_px is None:
        mask = circle_mask(image.shape, moon_center, mask_inner_radius_px)
    else:
        mask = masking.annulus_mask(image.shape, moon_center, mask_inner_radius_px, mask_outer_radius_px)
    image_for_alignment = mask * filtered_image
    return image_for_alignment
