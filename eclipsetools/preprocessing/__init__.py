import numpy as np

import eclipsetools.utils.circle_finder
from eclipsetools.preprocessing import filtering
from eclipsetools.preprocessing import masking
from eclipsetools.preprocessing.filtering import radial_high_pass_filter
from eclipsetools.preprocessing.masking import find_mask_radii_px, annulus_mask


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

    filtered_image = radial_high_pass_filter(image, moon.center)
    masked_image = filtered_image * annulus_mask(image.shape, moon.center, mask_inner_radius_px, mask_outer_radius_px)
    return masked_image
