import numpy as np
import cv2

def rotational_blur(
    image: np.ndarray,
    max_angle: float,
    center: tuple) -> np.ndarray:
    """    Apply a rotational blur to a grayscale image using polar coordinates.
    Parameters:
        image (np.ndarray): Input grayscale image (2D array).
        max_angle (float): Maximum rotation angle in degrees (blur extent).
        center (tuple): (x, y) coordinates of the rotation center.
    Returns:
        np.ndarray: Blurred image.
    """
    assert image.ndim == 2, "Input image must be grayscale (2D array)."

    # max_radius defines the outer boundary of the polar transform
    max_radius = np.sqrt(((image.shape[0] / 2.0) ** 2.0) + ((image.shape[1] / 2.0) ** 2.0))

    polar_image = cv2.linearPolar(
        src=image,
        center=(center[1], center[0]),
        maxRadius=max_radius,
        flags=cv2.WARP_POLAR_LINEAR)

    blur_width = int(max_angle * polar_image.shape[0] / 360.0)
    blur_kernel_size = (1, blur_width)

    blurred_polar = cv2.blur(src=polar_image, ksize=blur_kernel_size)

    return cv2.linearPolar(
        src=blurred_polar,
        center=(center[1], center[0]),
        maxRadius=max_radius,
        flags=cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP)
