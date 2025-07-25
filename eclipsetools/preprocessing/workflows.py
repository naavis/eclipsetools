import numpy as np

import eclipsetools.utils.circle_finder
from eclipsetools.preprocessing import filtering
from eclipsetools.preprocessing import masking
from eclipsetools.preprocessing.filtering import radial_high_pass_filter
from eclipsetools.preprocessing.masking import find_mask_radii_px, annulus_mask


def preprocess_with_auto_mask(
    rgb_image: np.ndarray,
    mask_inner_radius_multiplier: float,
    mask_outer_radius_multiplier: float,
    crop: int,
    moon_min_radius: int,
    moon_max_radius: int,
) -> np.ndarray:
    image = np.mean(rgb_image, axis=2, dtype=np.float32)
    moon = eclipsetools.utils.circle_finder.find_circle(
        image, moon_min_radius, moon_max_radius
    )
    assert (
        moon is not None
    ), "Moon not found in the image. Please check the input image."
    mask_inner_radius_px, mask_outer_radius_px = find_mask_radii_px(
        rgb_image, moon, mask_inner_radius_multiplier, mask_outer_radius_multiplier
    )

    crop_start = crop
    crop_end = -crop if crop > 0 else None
    filtered_image = radial_high_pass_filter(image, moon.center)[
        crop_start:crop_end, crop_start:crop_end
    ]
    masked_image = filtered_image * annulus_mask(
        filtered_image.shape,
        (moon.center[0] - crop, moon.center[1] - crop),
        mask_inner_radius_px,
        mask_outer_radius_px - crop,
    )
    return masked_image


def preprocess_with_fixed_mask(
    rgb_image: np.ndarray,
    mask_inner_radius_px: float,
    mask_outer_radius_multiplier: float,
    crop: int,
    moon_min_radius: int,
    moon_max_radius: int,
) -> np.ndarray:
    image = np.mean(rgb_image, axis=2, dtype=np.float32)
    moon = eclipsetools.utils.circle_finder.find_circle(
        image, moon_min_radius, moon_max_radius
    )
    assert (
        moon is not None
    ), "Moon not found in the image. Please check the input image."

    mask_outer_radius_px = mask_inner_radius_px * mask_outer_radius_multiplier
    crop_start = crop
    crop_end = -crop if crop > 0 else None
    filtered_image = filtering.radial_high_pass_filter(image, moon.center)[
        crop_start:crop_end, crop_start:crop_end
    ]
    masked_image = filtered_image * masking.annulus_mask(
        filtered_image.shape,
        (moon.center[0] - crop, moon.center[1] - crop),
        mask_inner_radius_px,
        mask_outer_radius_px - crop,
    )
    return masked_image
