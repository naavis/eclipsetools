import numpy as np

from eclipsetools.alignment.phase_correlation import phase_correlate_with_low_pass


def find_translation(ref_image, image, low_pass_sigma) -> np.ndarray:
    """
    Find the translation between two images using phase correlation.
    :param ref_image: Reference image to align against
    :param image: Image to be aligned
    :param low_pass_sigma: Standard deviation for Gaussian low-pass filter in frequency domain. The lower the value,
                           the stronger the low-pass filtering. If None, no low-pass filtering is applied.
    :return: Translation vector (dy, dx) indicating how much to shift image to align it with ref_image
    """
    return -np.array(phase_correlate_with_low_pass(ref_image, image, low_pass_sigma))
