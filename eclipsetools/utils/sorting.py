import joblib
import numpy as np
from tqdm import tqdm

from eclipsetools.utils.image_reader import open_image


def sort_images_by_brightness(
    image_paths: list[str],
    n_jobs: int = -1,
    n_samples: int = 10000,
    show_progress: bool = True,
) -> list[tuple[str, float]]:
    """
    Sort images by brightness level.

    :param image_paths: List of image file paths to analyze
    :param n_jobs: Number of parallel jobs for brightness estimation
    :param n_samples: Number of pixels to sample from each image for brightness estimation
    :param show_progress: Whether to show a progress bar
    :return: List of tuples (image_path, brightness) sorted by brightness (darkest first)
    """

    brightness_scores = tqdm(
        iterable=joblib.Parallel(
            n_jobs=n_jobs, prefer="threads", return_as="generator_unordered"
        )(
            joblib.delayed(_estimate_image_brightness)(image_path, n_samples)
            for image_path in image_paths
        ),
        total=len(image_paths),
        desc="Analyzing brightness",
        unit="img",
        disable=not show_progress,
    )

    # Sort by brightness (darkest first)
    return sorted(brightness_scores, key=lambda x: x[1])


def _estimate_image_brightness(image_path: str, n_samples: int) -> tuple[str, float]:
    """
    Estimate the brightness of an image.

    :param image_path: Path to the image file
    :param n_samples: Target number of pixels to sample for brightness estimation
    :return: Average brightness of sampled pixels
    """
    image = open_image(image_path)

    # Get image dimensions
    height, width = image.shape[:2]
    total_pixels = height * width

    # If image is smaller than sample size, use all pixels
    if total_pixels <= n_samples:
        return image_path, float(np.mean(image))

    # Calculate step size to get approximately n_samples pixels
    step = int(np.sqrt(total_pixels / n_samples))

    # Sample pixels using regular intervals
    sampled_pixels = image[::step, ::step]

    # Calculate mean brightness (average across all channels)
    return image_path, float(np.mean(sampled_pixels))
