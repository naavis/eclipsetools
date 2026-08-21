import cv2
import numpy as np

from eclipsetools.alignment.phase_correlation import phase_correlate_with_low_pass


def find_transform(
    ref_image: np.ndarray,
    image: np.ndarray,
    low_pass_sigma: float,
    allow_scale: bool,
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
             where dy, dx is the translation vector from the reference image to the aligned image.
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
    shift_y, shift_x = phase_correlate_with_low_pass(
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
    translation_y, translation_x = phase_correlate_with_low_pass(
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
