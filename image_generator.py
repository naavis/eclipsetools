import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

import eclipsetools.preprocessing
from eclipsetools.preprocessing.masking import hann_window_mask
from eclipsetools.utils.raw_reader import open_raw_image

SAVE_IMAGES = True


def main():
    ref_image = open_raw_image(r'tests\images\eclipse_5ms.CR3')

    crop_margin = 500

    ref_image_preproc = eclipsetools.preprocessing.preprocess_for_alignment(
        ref_image[crop_margin:-crop_margin, crop_margin:-crop_margin])
    ref_image_preproc = pad_with_zeros(ref_image_preproc)

    rotations = np.arange(0, 90 + 5, 5)
    for r in rotations:
        print(f'Expected rotation: {r} degrees')
        test_image = generate_test_image(ref_image, float(r))
        test_image_preproc = eclipsetools.preprocessing.preprocess_for_alignment(
            test_image[crop_margin:-crop_margin, crop_margin:-crop_margin])
        test_image_preproc = pad_with_zeros(test_image_preproc)
        test_image_filename = os.path.join('generator_output', f'test_image_rot_{r:02d}.png')
        if SAVE_IMAGES:
            plt.imsave(test_image_filename, test_image_preproc, cmap='gray')
        recovered_rotation, recovered_scale = find_rotation_scale(ref_image_preproc, test_image_preproc,
                                                                  suffix=f'rot_{r:02d}')
        print(f'Rotation: {recovered_rotation:.2f} degrees, Scale: {recovered_scale:.4f}')
        print('----------------------')


def pad_with_zeros(image: np.ndarray) -> np.ndarray:
    assert len(image.shape) == 2, "Input image must be a 2D array (grayscale image)."
    h, w = image.shape
    longer_side = max(h, w)
    padded = np.zeros((longer_side, longer_side), dtype=image.dtype)
    y_offset = (longer_side - h) // 2
    x_offset = (longer_side - w) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = image
    return padded


def generate_test_image(ref_image: np.ndarray, rotation: float) -> np.ndarray:
    """
    Generate a rotated version of the reference image.

    :param ref_image: Reference image to rotate
    :param rotation: Rotation angle in degrees
    :return: Rotated image
    """
    center = (ref_image.shape[1] // 2, ref_image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
    rotated_image = cv2.warpAffine(ref_image,
                                   rotation_matrix,
                                   dsize=(ref_image.shape[1], ref_image.shape[0]),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=[0, 0, 0]).astype(np.float32)
    return rotated_image


def find_rotation_scale(ref_image: np.ndarray, test_image: np.ndarray, suffix: str) -> tuple[float, float]:
    shape = ref_image.shape
    shortest_side = np.min(shape)
    polar_radius = shortest_side / 2.0

    # Find scale and rotation
    # We only need half of the log-polar FFTs, as they are symmetric
    ref_log_cart_filename = os.path.join('generator_output', f'fft_log_ref_cart_{suffix}.png')
    ref_log_polar_filename = os.path.join('generator_output', f'fft_log_ref_polar_{suffix}.png')
    ref_fft_log_polar = _log_polar_fft(ref_image, polar_radius, ref_log_cart_filename, ref_log_polar_filename)[
                        :shape[0] // 2, :]
    image_log_cart_filename = os.path.join('generator_output', f'fft_log_test_cart_{suffix}.png')
    image_log_polar_filename = os.path.join('generator_output', f'fft_log_test_polar_{suffix}.png')
    image_fft_log_polar = _log_polar_fft(test_image, polar_radius, image_log_cart_filename, image_log_polar_filename)[
                          :shape[0] // 2, :]

    # Find shifts in the log-polar FFTs using phase correlation
    shift_y, shift_x = _simple_phase_correlation(ref_fft_log_polar, image_fft_log_polar, suffix)

    # Recover rotation from the correlation result
    rotation_degrees = 360.0 * shift_y / (polar_radius * np.pi * np.pi)

    # Recover scale from the correlation result
    k_log = polar_radius / np.log(polar_radius)
    scale = np.exp(shift_x / k_log)

    return rotation_degrees, scale


def _log_polar_fft(image: np.ndarray, radius: float, cartesian_filename: str, polar_filename: str):
    assert len(image.shape) == 2, "Input image must be a 2D array (grayscale image)."
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    # Amplitude spectra have very high values in the low frequencies, so we use logarithm to compress the range
    fft_mag = np.log(1.0 + fft_mag)

    log_polar = cv2.warpPolar(
        src=fft_mag,
        dsize=(0, 0),
        center=(fft_mag.shape[1] / 2.0, fft_mag.shape[0] / 2.0),
        maxRadius=radius,
        flags=cv2.INTER_LINEAR | cv2.WARP_POLAR_LOG)

    if SAVE_IMAGES:
        plt.imsave(cartesian_filename, fft_mag, cmap='gray')
        plt.imsave(polar_filename, log_polar, cmap='gray')
    return log_polar


def _simple_phase_correlation(ref_image: np.ndarray, image: np.ndarray, suffix: str):
    assert ref_image.shape == image.shape, "Reference and image must have the same shape."

    window = hann_window_mask(ref_image.shape)

    fft1 = np.fft.fft2(window * ref_image)
    fft2 = np.fft.fft2(window * image)

    cross_power_spectrum = (fft1 * np.conj(fft2)) / (np.abs(fft1 * fft2) + 1e-8)
    phase_correlation = np.abs(np.fft.ifft2(cross_power_spectrum))

    if SAVE_IMAGES:
        plt.imsave(os.path.join('generator_output', f'phase_correlation_{suffix}.png'),
                   np.fft.fftshift(phase_correlation),
                   cmap='gray')

    shift_y, shift_x = np.unravel_index(np.argmax(phase_correlation), phase_correlation.shape)

    # Shifts peak coordinates to wrap around
    if shift_y > phase_correlation.shape[0] // 2:
        shift_y -= phase_correlation.shape[0]

    if shift_x > phase_correlation.shape[1] // 2:
        shift_x -= phase_correlation.shape[1]

    return shift_y, shift_x


if __name__ == "__main__":
    main()
