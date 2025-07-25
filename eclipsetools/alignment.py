import cv2
import numpy as np

from eclipsetools.preprocessing.masking import hann_window_mask


def find_translation(ref_image, image, low_pass_sigma) -> np.ndarray:
    """
    Find the translation between two images using phase correlation.
    :param ref_image: Reference image to align against
    :param image: Image to be aligned
    :param low_pass_sigma: Standard deviation for Gaussian low-pass filter in frequency domain.
        The larger the value, the more accurate the results are. But after a certain point, the accuracy collapses,
        and the reported translation is close to zero.
    :return: Translation vector (dy, dx) indicating how much the second image is shifted relative to the first.
    """
    return np.array(_phase_correlate_with_low_pass(ref_image, image, low_pass_sigma))


def _phase_correlate_with_low_pass(
    img_a: np.ndarray,
    img_b: np.ndarray,
    low_pass_sigma: float = None,
) -> np.ndarray:
    assert img_a.shape == img_b.shape

    window = hann_window_mask(img_a.shape)
    img_a_win = window * img_a
    img_b_win = window * img_b

    img_a_norm = (img_a_win - img_a_win.mean()) / img_a_win.std()
    img_b_norm = (img_b_win - img_b_win.mean()) / img_b_win.std()

    fft1 = np.fft.fft2(img_a_norm)
    fft2 = np.fft.fft2(img_b_norm)

    offset = 0.01 * np.max(np.abs(fft1))
    cross_power_spectrum = (
        fft1 * np.conjugate(fft2) / ((np.abs(fft1) + offset) * (np.abs(fft2) + offset))
    )

    if low_pass_sigma:
        gaussian_weighting = _gaussian_weights(
            cross_power_spectrum.shape, low_pass_sigma
        )
        phase_correlation = np.abs(
            np.fft.ifft2(gaussian_weighting * cross_power_spectrum)
        )
    else:
        phase_correlation = np.abs(np.fft.ifft2(cross_power_spectrum))

    phase_correlation = np.fft.ifftshift(phase_correlation)

    initial_peak = np.unravel_index(np.argmax(phase_correlation), img_a.shape)
    subpixel_peak = _center_of_mass(
        phase_correlation,
        (int(initial_peak[0]), int(initial_peak[1])),
        5,
    )

    return -(subpixel_peak - np.array(img_a.shape) // 2)


def find_transform(
    ref_image,
    image,
    low_pass_sigma,
    allow_scale: bool = True,
) -> tuple[float, float, tuple[float, float]]:
    """
    Find the scale, rotation, and translation between two images.

    This function uses a multistep approach:
    1. Find scale and rotation using log-polar transform and phase correlation
    2. Apply the found scale and rotation to the image
    3. Find the translation using phase correlation on the transformed image

    :param ref_image: Reference image to align against
    :param image: Image to be aligned
    :param low_pass_sigma: Standard deviation for Gaussian low-pass filter in frequency domain when finding translation.
    :param allow_scale: Allow estimating scale. If False, scale is assumed to be 1.0.
    :return: Tuple containing (scale, rotation_angle_degrees, (dy, dx))
             where dy, dx is the translation vector
    """

    ref_image_pad = _pad_with_zeros(ref_image)
    image_pad = _pad_with_zeros(image)

    shape = ref_image_pad.shape
    shortest_side = np.min(shape)
    radius = shortest_side // 8

    # Step 1: Find scale and rotation
    # We only need half of the log-polar FFTs, as they are symmetric
    ref_fft_log_polar = _log_polar_fft(ref_image_pad, radius)[: shape[0] // 2, :]
    image_fft_log_polar = _log_polar_fft(image_pad, radius)[: shape[0] // 2, :]

    # Find shifts in the log-polar FFTs using phase correlation
    shift_y, shift_x = _phase_correlate_with_low_pass(
        ref_fft_log_polar, image_fft_log_polar
    )

    # Recover rotation from the correlation result
    rotation_degrees = -360.0 * shift_y / ref_fft_log_polar.shape[0]

    # Recover scale from the correlation result
    k_log = radius / np.log(radius)
    scale = np.exp(-shift_x / k_log) if allow_scale else 1.0

    # Step 2: Apply scale and rotation to the image
    rotate_scale_matrix = cv2.getRotationMatrix2D(
        (image_pad.shape[1] // 2, image_pad.shape[0] // 2),
        -rotation_degrees,
        1.0 / scale,
    )

    translated_image = cv2.warpAffine(
        image_pad,
        rotate_scale_matrix,
        dsize=(image_pad.shape[1], image_pad.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=[0, 0, 0],
    ).astype(np.float32)

    # Step 3: Find translation between reference and transformed image
    translation_y, translation_x = find_translation(
        ref_image_pad, translated_image, low_pass_sigma
    )

    return (
        float(scale),
        float(rotation_degrees),
        (float(translation_y), float(translation_x)),
    )


def _pad_with_zeros(image: np.ndarray) -> np.ndarray:
    assert len(image.shape) == 2, "Input image must be a 2D array (grayscale image)."
    h, w = image.shape
    longer_side = max(h, w)
    padded = np.zeros((longer_side, longer_side), dtype=image.dtype)
    y_offset = (longer_side - h) // 2
    x_offset = (longer_side - w) // 2
    padded[y_offset : y_offset + h, x_offset : x_offset + w] = image
    return padded


def _log_polar_fft(image: np.ndarray, radius: float) -> np.ndarray:
    assert len(image.shape) == 2, "Input image must be a 2D array (grayscale image)."
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    # Amplitude spectra have very high values in the low frequencies, so we use logarithm to compress the range
    fft_mag = np.log(1.0 + fft_mag)
    log_polar = cv2.warpPolar(
        src=fft_mag,
        dsize=(0, 0),
        center=(fft_mag.shape[1] // 2, fft_mag.shape[0] // 2),
        maxRadius=radius,
        flags=cv2.INTER_LINEAR | cv2.WARP_POLAR_LOG,
    )
    return log_polar


def _gaussian_weights(shape: tuple, sigma: float) -> np.ndarray:
    fy = np.fft.fftfreq(shape[1])
    fx = np.fft.fftfreq(shape[0])
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
