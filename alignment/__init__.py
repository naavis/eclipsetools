import numpy as np


def find_translation(ref_image, image):
    fft1 = np.fft.fft2(ref_image)
    fft2 = np.fft.fft2(image)
    offset = 0.01
    cross_power_spectrum = fft1 * np.conjugate(fft2) / ((np.abs(fft1) + offset) * (np.abs(fft2) + offset))

    # TODO: The sigma parameter Gaussian weighting should be adjustable
    sigma = 0.1 * cross_power_spectrum.shape[1]
    gaussian_weighting = np.exp(-0.5 * np.sum(np.square(np.indices(cross_power_spectrum.shape)), axis=0) / (sigma ** 2))

    phase_correlation = np.real(np.fft.ifft2(gaussian_weighting * cross_power_spectrum))
    peak = np.unravel_index(np.argmax(phase_correlation), image.shape)

    peak = _center_of_mass(phase_correlation, peak, 2)

    # Upper half of each axis represents negative translations
    thresholds = np.array(ref_image.shape[:1]) // 2
    subtractions = np.array(ref_image.shape[:1])
    peak = np.where(peak > thresholds, peak - subtractions, peak)

    return -peak


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
