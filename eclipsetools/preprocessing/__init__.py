import numpy as np

import eclipsetools.utils.circle_finder
from eclipsetools.preprocessing import filtering
from eclipsetools.preprocessing import masking
from eclipsetools.preprocessing.masking import circle_mask


def preprocess_for_alignment(rgb_image: np.ndarray, mask_inner_radius: float, mask_outer_radius: float) -> np.ndarray:
    image = np.mean(rgb_image, axis=2, dtype=np.float32)
    moon = eclipsetools.utils.circle_finder.find_circle(image, min_radius=400, max_radius=600)
    assert moon is not None, "Moon not found in the image. Please check the input image."

    saturated_radius = masking.estimate_saturated_radius(moon, rgb_image)

    mask_inner_radius_px = mask_inner_radius * (saturated_radius if saturated_radius else moon.radius)
    mask_outer_radius_px = mask_outer_radius * mask_inner_radius_px

    preproc = _mask_and_filter(image, moon.center, mask_inner_radius_px, mask_outer_radius_px)
    return preproc


def _mask_and_filter(image: np.ndarray, moon_center: tuple, mask_inner_radius_px: float,
                     mask_outer_radius_px: float) -> np.ndarray:
    # TODO: Parametrize sigma, which is used to control the amount of rotational blur used in the tangential high-pass filter
    blurred_image = filtering.rotational_blur(image,
                                              sigma=2.0,
                                              center=moon_center)
    filtered_image = image - blurred_image

    # We limit the outer radius of the annulus mask to avoid intersecting the image edges.
    shortest_distance_to_edge = min(moon_center[0],
                                    moon_center[1],
                                    image.shape[0] - moon_center[0],
                                    image.shape[1] - moon_center[1])
    mask_outer_radius = min(mask_outer_radius_px, shortest_distance_to_edge * 0.95)

    if mask_outer_radius_px > 0.0:
        mask = masking.annulus_mask(image.shape, moon_center, mask_inner_radius_px, mask_outer_radius)
    else:
        mask = circle_mask(image.shape, moon_center, mask_inner_radius_px)
    image_for_alignment = mask * filtered_image
    return image_for_alignment
