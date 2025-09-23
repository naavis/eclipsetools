import numba
import numpy as np
from numba_progress import ProgressBar


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

                result[i, j] = _fit_plane_and_recover_pixel(image_region, mask_region)
        progress_proxy.update(1)
    return result


@numba.jit(nogil=True)
def _fit_plane_and_recover_pixel(
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
