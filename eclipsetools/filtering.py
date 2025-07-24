import numba
import numpy as np
from numba_progress import ProgressBar


def get_kernel_size(sigma_tangent: float, sigma_radial: float) -> int:
    """
    Calculate the kernel size based on the maximum of the two sigmas.
    The kernel size is always odd and at least 3.
    """
    max_sigma = max(sigma_tangent, sigma_radial)
    kernel_size = int(max_sigma * 4) | 1  # Ensure it's odd
    return max(kernel_size, 3)  # Ensure minimum size of 3


@numba.jit(nogil=True)
def inpaint_pixels(
    image: np.ndarray,
    infill_mask: np.ndarray,
    image_mask: np.ndarray,
    kernel_size: int,
    progress_proxy: ProgressBar,
) -> np.ndarray:
    # TODO: Write documentation for this function
    result = image.copy()
    half_kernel = kernel_size // 2
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if infill_mask[i, j]:
                # Define the region around the pixel to inpaint
                start_i = max(0, i - half_kernel)
                end_i = min(image.shape[0], i + half_kernel + 1)
                start_j = max(0, j - half_kernel)
                end_j = min(image.shape[1], j + half_kernel + 1)

                # Extract the region and compute the mean of unmasked pixels
                image_region = image[start_i:end_i, start_j:end_j]
                mask_region = image_mask[start_i:end_i, start_j:end_j]

                result[i, j] = fit_plane_and_recover_pixel(image_region, mask_region)
        progress_proxy.update(1)
    return result


@numba.jit(nogil=True)
def fit_plane_and_recover_pixel(
    image_region: np.ndarray, mask_region: np.ndarray
) -> float:
    """
    Fit a plane to unmasked pixels and return the value at the center pixel.

    :param image_region: 2D array of image values
    :param mask_region: 2D array where True = unmasked, False = masked
    :return: Plane value at center pixel, or mean of unmasked pixels if plane fitting fails
    """
    h, w = image_region.shape
    center_i = h // 2
    center_j = w // 2

    # Collect unmasked pixel coordinates and values
    unmasked_coords = []
    unmasked_values = []

    for i in range(h):
        for j in range(w):
            if mask_region[i, j]:  # True for unmasked pixels
                unmasked_coords.append((i, j))
                unmasked_values.append(image_region[i, j])

    # Need at least 3 points to fit a plane
    if len(unmasked_coords) < 3:
        # Fallback: return mean of available unmasked pixels
        if len(unmasked_coords) > 0:
            return float(np.mean(np.array(unmasked_values)))
        else:
            return 0.0

    # Set up linear system: z = ax + by + c
    # Convert to matrix form: coef_array * params = b
    n_points = len(unmasked_coords)
    coef_array = np.zeros((n_points, 3))
    b = np.zeros(n_points)

    for idx in range(n_points):
        i, j = unmasked_coords[idx]
        coef_array[idx, 0] = i  # coefficient for 'a'
        coef_array[idx, 1] = j  # coefficient for 'b'
        coef_array[idx, 2] = 1  # coefficient for 'c'
        b[idx] = unmasked_values[idx]

    # Numba doesn't support catching specific exceptions,
    # so we suppress warnings about too broad exception catching
    # noinspection PyBroadException
    try:
        params = np.linalg.lstsq(coef_array, b)
        solved_a, solved_b, solved_c = params[0]

        # Return plane value at center pixel
        return solved_a * center_i + solved_b * center_j + solved_c
    except:
        # If throws, fallback to mean of unmasked pixels
        return float(np.mean(np.array(unmasked_values)))


def partial_convolution(
    image: np.ndarray,
    mask: np.ndarray,
    sigma_tangent: float,
    sigma_radial: float,
    center: tuple[float, float],
) -> np.ndarray:
    # TODO: Write documentation for this function
    kernel_size = get_kernel_size(sigma_tangent, sigma_radial)
    padding = kernel_size // 2

    padded_image = np.pad(image, padding, mode="constant", constant_values=0)
    padded_mask = np.pad(mask, padding, mode="constant", constant_values=0)

    padded_r_grid, padded_theta_grid = compute_polar_grid(
        padded_image.shape, (center[0] + padding, center[1] + padding)
    )
    with ProgressBar(
        total=image.shape[0], unit="row", desc="Convolving image"
    ) as progress_proxy:
        return partial_convolution_loop(
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


def compute_polar_grid(
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
def partial_convolution_loop(
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
    # TODO: Write documentation for this function
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


@numba.jit(nogil=True)
def adaptive_kernel(
    r_grid: np.ndarray,
    theta_grid: np.ndarray,
    sigma_tangent: float,
    sigma_radial: float,
) -> np.ndarray:
    """
    Create an adaptive kernel based on the distance from the center and angle.
    The kernel size is deduced from r_grid size.
    """
    cx = r_grid.shape[1] // 2
    cy = r_grid.shape[0] // 2
    cr = r_grid[cy, cx]  # Center radius
    ct = theta_grid[cy, cx]  # Center angle

    # Calculate circular angular distance (handles 0/2π wraparound)
    theta_diff = theta_grid - ct
    theta_diff = np.minimum(np.abs(theta_diff), 2 * np.pi - np.abs(theta_diff))

    exponent = np.zeros_like(r_grid, dtype=np.float32)
    # Tiny sigma values can cause artifacts, and an unsharp mask of less than 0.1 does not make any practical sense
    if sigma_radial > 0.1:
        exponent += -((r_grid - cr) ** 2) / (2 * sigma_radial**2)
    if sigma_tangent > 0.1:
        exponent += -((r_grid * theta_diff) ** 2) / (2 * sigma_tangent**2)

    return np.exp(exponent).astype(np.float32)
