import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def radial_high_pass_filter(image: np.ndarray, moon_center: tuple) -> np.ndarray:
    blurred_image = _rotational_blur(image, sigma=2.0, center=moon_center)
    return image - blurred_image


def _rotational_blur(image: np.ndarray, sigma: float, center: tuple) -> np.ndarray:
    """
    Apply a rotational Gaussian blur to a grayscale image using polar coordinates.
    :param image: Input grayscale image (2D array).
    :param sigma: Blur sigma in degrees.
    :param center: (y, x) coordinates of the rotation center.
    :return: Blurred image as a 2D array.
    """
    assert image.ndim == 2, "Input image must be grayscale (2D array)."

    # max_radius defines the outer boundary of the polar transform
    max_radius = np.sqrt((image.shape[0] ** 2.0) + (image.shape[1] ** 2.0))

    polar_image = cv2.linearPolar(
        src=image,
        center=(center[1], center[0]),
        maxRadius=max_radius,
        flags=cv2.INTER_LINEAR | cv2.WARP_POLAR_LINEAR,
    )

    sigma_pixels = sigma * polar_image.shape[0] / 360.0
    blurred_polar = gaussian_filter(
        polar_image, sigma=sigma_pixels, mode="wrap", axes=(0,)
    )

    return cv2.linearPolar(
        src=blurred_polar,
        center=(center[1], center[0]),
        maxRadius=max_radius,
        flags=cv2.INTER_LINEAR | cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP,
    )
