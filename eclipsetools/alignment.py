import cv2
import numpy as np
from matplotlib import pyplot as plt


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

    phase_correlation = np.abs(np.fft.ifft2(gaussian_weighting * cross_power_spectrum))

    initial_peak = np.unravel_index(np.argmax(phase_correlation), image.shape)
    subpixel_peak = _center_of_mass(phase_correlation, initial_peak, 4)

    # Upper half of each axis represents negative translations
    thresholds = np.array(phase_correlation.shape[:2]) // 2
    subtractions = np.array(phase_correlation.shape[:2])
    subpixel_peak = np.where(subpixel_peak > thresholds, subpixel_peak - subtractions, subpixel_peak)

    return -subpixel_peak


def find_transform(ref_image, image, low_pass_sigma):
    """
    Find the scale, rotation, and translation between two images.

    This function uses a multi-step approach:
    1. Find scale and rotation using log-polar transform and phase correlation
    2. Apply the found scale and rotation to the image
    3. Find the translation using phase correlation on the transformed image

    :param ref_image: Reference image to align against
    :param image: Image to be aligned
    :param low_pass_sigma: Standard deviation for Gaussian low-pass filter in frequency domain when finding translation.
    :return: Tuple containing (scale, rotation_angle_degrees, (dy, dx))
             where dy, dx is the translation vector
    """
    shortest_side = np.min(ref_image.shape)
    radius = shortest_side // 2

    # Step 1: Find scale and rotation
    ref_log_polar = _log_polar_fft(ref_image, radius)
    transformed_log_polar = _log_polar_fft(image, radius)

    # We only need half of the log-polar FFTs, as they are symmetric
    ref_log_polar = ref_log_polar[:ref_log_polar.shape[0] // 2, :]
    transformed_log_polar = transformed_log_polar[:transformed_log_polar.shape[0] // 2, :]

    fix, ax = plt.subplots(1, 2)
    ax[0].imshow(ref_log_polar, cmap='gray')
    ax[0].set_title('Reference Log-Polar FFT')
    ax[1].imshow(transformed_log_polar, cmap='gray')
    ax[1].set_title('Transformed Log-Polar FFT')
    plt.show()

    # Recover rotation from the correlation result
    correlation_result = cv2.phaseCorrelate(ref_log_polar, transformed_log_polar)
    rotation_degrees = 360.0 * -correlation_result[0][1] / ref_log_polar.shape[0]

    # Recover scale from the correlation result
    klog = shortest_side / np.log(radius)
    scale = np.exp(-correlation_result[0][0] / klog)

    # Step 2: Apply scale and rotation to the image
    derotate_rescale_matrix = cv2.getRotationMatrix2D(
        (image.shape[1] // 2, image.shape[0] // 2), -rotation_degrees, 1.0 / scale)

    translated_image = cv2.warpAffine(
        image,
        derotate_rescale_matrix,
        dsize=(image.shape[1], image.shape[0]),
        borderMode=cv2.BORDER_REFLECT_101)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].set_title('Before derotation and rescaling')
    ax[0, 1].imshow(translated_image, cmap='gray')
    ax[0, 1].set_title('After derotation and rescaling')
    ax[1, 0].imshow(ref_image, cmap='gray')
    ax[1, 0].set_title('Reference image')
    plt.show()

    # Step 3: Find translation between reference and transformed image
    translation = find_translation(ref_image, translated_image, low_pass_sigma)

    return scale, rotation_degrees, translation


def _log_polar_fft(image, radius):
    assert len(image.shape) == 2, "Input image must be a 2D array (grayscale image)."
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    # Amplitude spectra have very high values in the low frequencies, so we use logarithm to compress the range
    fft_mag = np.log(1.0 + fft_mag)
    log_polar = cv2.warpPolar(
        src=fft_mag,
        dsize=(fft_mag.shape[1], fft_mag.shape[0]),
        center=(fft_mag.shape[1] // 2, fft_mag.shape[0] // 2),
        maxRadius=radius,
        flags=cv2.INTER_LINEAR | cv2.WARP_POLAR_LOG)
    return log_polar


def _gaussian_weights(shape, sigma):
    fy = np.fft.fftfreq(shape[1])
    fx = np.fft.fftfreq(shape[0])
    fy_grid, fx_grid = np.meshgrid(fy, fx)
    freq_squared = fy_grid ** 2 + fx_grid ** 2
    gaussian_weighting = np.exp(-0.5 * freq_squared / (sigma ** 2), dtype=np.float32)
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
    y_com_global = np.float32((y0 - half_size + y_com_local) % height)
    x_com_global = np.float32((x0 - half_size + x_com_local) % width)

    return y_com_global, x_com_global
