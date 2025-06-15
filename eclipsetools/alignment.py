import numpy as np


def find_translation(ref_image, image, low_pass_sigma):
    """
    Find the translation between two images using phase correlation.
    :param ref_image: Reference image to align against
    :param image: Image to be aligned
    :param low_pass_sigma: Standard deviation for Gaussian low-pass filter in frequency domain.
        The larger the value, the more accurate the results are. But after a certain point, the accuracy collapses,
        and the reported translation is close to zero.
    :return: Translation vector (dy, dx) indicating how much the second image is shifted relative to the first.
    """
    fft1 = np.fft.fft2(ref_image)
    fft2 = np.fft.fft2(image)
    offset = 0.01 * np.max(np.abs(fft1))
    cross_power_spectrum = fft1 * np.conjugate(fft2) / ((np.abs(fft1) + offset) * (np.abs(fft2) + offset))

    gaussian_weighting = _gaussian_weights(cross_power_spectrum.shape, low_pass_sigma)

    phase_correlation = np.real(np.fft.ifft2(gaussian_weighting * cross_power_spectrum))

    initial_peak = np.unravel_index(np.argmax(phase_correlation), image.shape)
    subpixel_peak = _center_of_mass(phase_correlation, initial_peak, 4)

    # Upper half of each axis represents negative translations
    thresholds = np.array(phase_correlation.shape[:2]) // 2
    subtractions = np.array(phase_correlation.shape[:2])
    subpixel_peak = np.where(subpixel_peak > thresholds, subpixel_peak - subtractions, subpixel_peak)

    return -subpixel_peak


def _gaussian_weights(shape, sigma):
    fy = np.fft.fftfreq(shape[1])
    fx = np.fft.fftfreq(shape[0])
    fy_grid, fx_grid = np.meshgrid(fy, fx)
    freq_squared = fy_grid ** 2 + fx_grid ** 2
    gaussian_weighting = np.exp(-0.5 * freq_squared / (sigma ** 2))
    return gaussian_weighting


def _center_of_mass(image, center_point, window_size):
    height, width = image.shape
    y0, x0 = center_point
    half_size = window_size // 2

    # Generate wrapped indices
    y_indices = (np.arange(y0 - half_size, y0 + half_size + 1) % height)
    x_indices = (np.arange(x0 - half_size, x0 + half_size + 1) % width)

    # Extract wrapped region
    region = image[np.ix_(y_indices, x_indices)]

    # Create coordinate grid (local to the region)
    y_grid, x_grid = np.indices(region.shape)

    # Compute total intensity
    total_mass = region.sum()
    if total_mass == 0:
        return y0, x0  # or None

    # Compute local CoM
    y_com_local = (y_grid * region).sum() / total_mass
    x_com_local = (x_grid * region).sum() / total_mass

    # Map back to global coordinates (using wrapping)
    y_com_global = (y0 - half_size + y_com_local) % height
    x_com_global = (x0 - half_size + x_com_local) % width

    return y_com_global, x_com_global
