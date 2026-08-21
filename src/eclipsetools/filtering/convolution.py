import numba
import numpy as np
from numba_progress import ProgressBar

from eclipsetools.filtering.kernels import get_kernel_size, adaptive_kernel


def partial_convolution(
    image: np.ndarray,
    mask: np.ndarray,
    sigma_tangent: float,
    sigma_radial: float,
    center: tuple[float, float],
) -> np.ndarray:
    """
    Perform partial convolution on the image using an adaptive Gaussian kernel defined in polar coordinates.
    The convolution is only applied to pixels where the mask is True. Masked pixels remain unchanged.
    :param image: 2D array of image values
    :param mask: 2D boolean array where True = pixel to convolve, False = pixel to leave unchanged
    :param sigma_tangent: Gaussian convolution kernel sigma in the tangential direction (along circles centered at 'center')
    :param sigma_radial: Gaussian convolution kernel sigma in the radial direction (along rays from 'center')
    :param center: (cy, cx) pixel coordinates of the center for polar coordinate transformation
    :return: Partially convolved image as a 2D array
    """
    kernel_size = get_kernel_size(sigma_tangent, sigma_radial)
    padding = kernel_size // 2

    padded_image = np.pad(image, padding, mode="constant", constant_values=0)
    padded_mask = np.pad(mask, padding, mode="constant", constant_values=0)

    padded_r_grid, padded_theta_grid = _compute_polar_grid(
        padded_image.shape, (center[0] + padding, center[1] + padding)
    )
    with ProgressBar(
        total=image.shape[0], unit="row", desc="Convolving image"
    ) as progress_proxy:
        return _partial_convolution_loop(
            image,
            kernel_size,
            padded_image,
            padded_mask,
            padded_r_grid,
            padded_theta_grid,
            sigma_tangent,
            sigma_radial,
            progress_proxy,
        )


def _compute_polar_grid(
    image_shape: tuple[int, ...], center: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute radius and angle (polar coordinates) for every pixel in a 2D array.

    :param image_shape: (height, width) of the image
    :param center: (cy, cx) pixel coordinates of the center
    :return: r_grid - 2D array of radii, theta_grid - 2D array of angles in radians, [0, 2π)
    """
    height, width = image_shape
    cy, cx = center

    # Create grid of (x, y) coordinates
    y_indices, x_indices = np.indices((height, width))

    # Cartesian offsets from center
    dx = x_indices - cx
    dy = y_indices - cy

    # Polar coordinates
    r_grid = np.sqrt(dx**2 + dy**2, dtype=np.float32)
    theta_grid = np.arctan2(dy, dx, dtype=np.float32)
    theta_grid = np.mod(theta_grid, 2 * np.pi, dtype=np.float32)  # Normalize to [0, 2π)

    return r_grid, theta_grid


@numba.jit(nogil=True, parallel=True)
def _partial_convolution_loop(
    image: np.ndarray,
    kernel_size: int,
    padded_image: np.ndarray,
    padded_mask: np.ndarray,
    padded_r_grid: np.ndarray,
    padded_theta_grid: np.ndarray,
    sigma_tangent: float,
    sigma_radial: float,
    progress_proxy: ProgressBar,
) -> np.ndarray:
    half_kernel = kernel_size // 2

    padding = (padded_image.shape[0] - image.shape[0]) // 2

    result = image.copy()
    for i in numba.prange(image.shape[0]):
        for j in range(image.shape[1]):
            pad_i = i + padding
            pad_j = j + padding

            # If pixel is masked, skip convolution entirely
            if not padded_mask[pad_i, pad_j]:
                continue

            slice_start_i = pad_i - half_kernel
            slice_end_i = pad_i + half_kernel + 1
            slice_start_j = pad_j - half_kernel
            slice_end_j = pad_j + half_kernel + 1

            kernel = adaptive_kernel(
                padded_r_grid[slice_start_i:slice_end_i, slice_start_j:slice_end_j],
                padded_theta_grid[slice_start_i:slice_end_i, slice_start_j:slice_end_j],
                sigma_tangent,
                sigma_radial,
            )

            mask_region = padded_mask[
                slice_start_i:slice_end_i, slice_start_j:slice_end_j
            ]
            image_region = padded_image[
                slice_start_i:slice_end_i, slice_start_j:slice_end_j
            ]

            convolved_sum = np.sum(kernel * mask_region * image_region)
            weights = np.sum(kernel * mask_region)
            result[i, j] = (
                convolved_sum / weights if weights > 0 else padded_image[pad_i, pad_j]
            )
        progress_proxy.update(1)
    return result
