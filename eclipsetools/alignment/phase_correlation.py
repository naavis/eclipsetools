import numpy as np

from eclipsetools.preprocessing.masking import hann_window_mask


def phase_correlate_with_low_pass(
    img_a: np.ndarray,
    img_b: np.ndarray,
    low_pass_sigma: float | None = None,
) -> np.ndarray:
    """
    Phase correlate two images with optional low-pass filtering in the frequency domain. This method was suggested
    by Druckmüller and Druckmüllerová. The correlation peak is further refined by calculating the center of mass
    in a 5x5 window around the initial peak.
    :param img_a: First image
    :param img_b: Second image
    :param low_pass_sigma: Standard deviation for Gaussian low-pass filter in frequency domain. The lower the value,
                           the stronger the low-pass filtering. If None, no low-pass filtering is applied.
    :return: Translation vector (dy, dx) indicating how much the second image is shifted relative to the first.
    """
    assert img_a.shape == img_b.shape

    window = hann_window_mask(img_a.shape)
    img_a_win = window * img_a
    img_b_win = window * img_b

    img_a_norm = (img_a_win - img_a_win.mean()) / img_a_win.std()
    img_b_norm = (img_b_win - img_b_win.mean()) / img_b_win.std()

    fft1 = np.fft.rfft2(img_a_norm)
    fft2 = np.fft.rfft2(img_b_norm)

    offset = 0.01 * np.max(np.abs(fft1))
    cross_power_spectrum = (
        fft1 * np.conjugate(fft2) / ((np.abs(fft1) + offset) * (np.abs(fft2) + offset))
    )

    if low_pass_sigma:
        gaussian_weighting = _gaussian_weights(low_pass_sigma, img_a.shape)
        phase_correlation = np.abs(
            np.fft.irfft2(gaussian_weighting * cross_power_spectrum, s=img_a.shape)
        )
    else:
        phase_correlation = np.abs(np.fft.irfft2(cross_power_spectrum, s=img_a.shape))

    phase_correlation = np.fft.ifftshift(phase_correlation)

    initial_peak = np.unravel_index(np.argmax(phase_correlation), img_a.shape)
    subpixel_peak = _center_of_mass(
        phase_correlation,
        (int(initial_peak[0]), int(initial_peak[1])),
        5,
    )

    return np.array(img_a.shape) / 2 - subpixel_peak


def _gaussian_weights(sigma: float, image_shape: tuple) -> np.ndarray:
    fy = np.fft.rfftfreq(image_shape[1])
    fx = np.fft.fftfreq(image_shape[0])
    fy_grid, fx_grid = np.meshgrid(fy, fx)
    freq_squared = fy_grid**2 + fx_grid**2
    gaussian_weighting = np.exp(-0.5 * freq_squared / (sigma**2), dtype=np.float32)
    return gaussian_weighting


def _center_of_mass(
    image: np.ndarray,
    center_point: tuple[int, int],
    window_size: int,
) -> tuple[float, float]:
    height, width = image.shape
    y0, x0 = center_point
    half_size = window_size // 2

    # Generate wrapped indices
    y_indices = np.arange(y0 - half_size, y0 + half_size + 1) % height
    x_indices = np.arange(x0 - half_size, x0 + half_size + 1) % width

    # Extract wrapped region
    region = image[np.ix_(y_indices, x_indices)]

    # Create coordinate grid (local to the region)
    y_grid, x_grid = np.indices(region.shape)

    # Compute total intensity
    total_mass = region.sum()
    if total_mass == 0:
        return float(y0), float(x0)  # or None

    # Compute local CoM
    y_com_local = (y_grid * region).sum() / total_mass
    x_com_local = (x_grid * region).sum() / total_mass

    # Map back to global coordinates (using wrapping)
    y_com_global = np.float32((y0 - half_size + y_com_local) % height)
    x_com_global = np.float32((x0 - half_size + x_com_local) % width)

    return y_com_global, x_com_global
