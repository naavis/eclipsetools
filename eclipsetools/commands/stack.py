from pathlib import Path

import click
import joblib
import numpy as np
from tqdm import tqdm

from eclipsetools.stacking import linear_fit, weight_function_sigmoid
from eclipsetools.utils.circle_finder import find_circle, DetectedCircle
from eclipsetools.utils.image_reader import open_image
from eclipsetools.utils.image_writer import save_tiff


@click.command()
@click.argument("reference_image", type=click.Path(exists=True))
@click.argument("images_to_stack", nargs=-1, required=True)
@click.option(
    "--n-jobs",
    default=-1,
    type=int,
    help="Number of parallel jobs. Default is -1 (all CPUs).",
)
@click.option(
    "--output-file",
    type=click.Path(),
    default="hdr_stacked_image.tiff",
    help="Output filename for the stacked image tiff file.",
)
def hdr_stack(
    reference_image: str, images_to_stack: tuple[str], n_jobs: int, output_file: str
):
    """
    Stack multiple images to HDR stack. Images must be pre-aligned. Images taken with different exposure times
    are combined by linear fitting to the reference image.
    """
    ref_image = open_image(reference_image)
    ref_image -= min(ref_image.min(), 0.0)  # Ensure the reference image is non-negative

    # TODO: Parametrize mask sizes
    # Mask out moving moon, because it confuses the linear fitting
    linear_fit_mask_size = 1.1
    # TODO: Parametrize moon size range
    ref_moon_params = find_circle(ref_image[:, :, 1], min_radius=400, max_radius=600)
    ref_linear_fit_moon_mask = _get_moon_mask(
        ref_image.shape, ref_moon_params, linear_fit_mask_size
    )
    # Use a different size mask for the weighting function to make have a clean moon edge
    weighting_mask_size = 1.025
    ref_weighting_moon_mask = _get_moon_mask(
        ref_image.shape, ref_moon_params, weighting_mask_size
    )

    total_weights = np.zeros_like(ref_image)
    weighted_sum = np.zeros_like(ref_image)

    for image in tqdm(
        iterable=joblib.Parallel(
            n_jobs=n_jobs, prefer="threads", return_as="generator_unordered"
        )(
            joblib.delayed(_hdr_process_image)(
                reference_image,
                ref_image,
                ref_linear_fit_moon_mask,
                linear_fit_mask_size,
                ref_weighting_moon_mask,
                weighting_mask_size,
                image_path,
            )
            for image_path in images_to_stack
        ),
        total=len(images_to_stack),
        desc="Combining images",
        unit="img",
    ):
        weights, calibrated_image = image
        weighted_sum += weights * calibrated_image
        total_weights += weights

    stacked_image = weighted_sum / total_weights

    # Ensure there are no negative values, and normalize values to maximum of 1.0
    stacked_image -= min(stacked_image.min(), 0.0)
    stacked_image /= max(stacked_image.max(), 1.0)
    click.echo(f"Saving stacked image to {output_file}")
    save_tiff(stacked_image, output_file)


def _hdr_process_image(
    ref_image_path: str,
    ref_image: np.ndarray,
    ref_linear_fit_moon_mask: np.ndarray,
    linear_fit_mask_size: float,
    ref_weighting_moon_mask: np.ndarray,
    weighting_mask_size: float,
    image_path: str,
):
    image = open_image(image_path)
    assert image.shape == ref_image.shape, "Stacked images must have the same shape"
    is_reference = Path(image_path).resolve() == Path(ref_image_path).resolve()
    # TODO: Parametrize moon size range
    image_moon_params = find_circle(image[:, :, 1], min_radius=400, max_radius=600)
    image_linear_fit_moon_mask = _get_moon_mask(
        image.shape, image_moon_params, linear_fit_mask_size
    )
    # We linear fit only pixels that are not contaminated by the moon in either image.
    pixels_without_moon = ref_linear_fit_moon_mask & image_linear_fit_moon_mask
    ref_linear_fit_points = ref_image[pixels_without_moon, :].ravel()
    image_linear_fit_points = image[pixels_without_moon, :].ravel()
    linear_coef, linear_intercept = linear_fit(
        ref_linear_fit_points, image_linear_fit_points
    )
    weights = weight_function_sigmoid(image)
    if not is_reference:
        image_weighting_moon_mask = _get_moon_mask(
            image.shape, image_moon_params, weighting_mask_size
        )
        pixels_with_moon = ~ref_weighting_moon_mask | ~image_weighting_moon_mask
        weights[pixels_with_moon] = 0.0
    calibrated_image = (image - linear_intercept) / linear_coef
    return weights, calibrated_image


def _get_moon_mask(
    shape: tuple, moon_params: DetectedCircle, mask_size: float
) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    distances = np.sqrt(
        (x - moon_params.center[1]) ** 2 + (y - moon_params.center[0]) ** 2
    )
    moon_mask = distances >= mask_size * moon_params.radius
    return moon_mask


@click.command()
@click.argument("images_to_stack", nargs=-1, required=True)
@click.option(
    "--output-file",
    type=click.Path(),
    default="average_stacked_image.tiff",
    help="Output filename for the stacked image tiff file.",
)
def average_stack(images_to_stack: list[str], output_file: str):
    """
    Stack multiple images by averaging them together. This is useful for images taken with the same exposure time.
    """
    stacked_image = None
    counts = None
    for image_path in tqdm(iterable=images_to_stack, desc="Stacking images"):
        image = open_image(image_path)
        has_values = image > 0.0
        if stacked_image is None:
            stacked_image = image
            counts = np.ones_like(image, dtype=np.uint32)
        else:
            stacked_image += image
            counts += has_values.astype(np.uint32)

    stacked_image /= counts
    click.echo(f"Saving stacked image to {output_file}")
    save_tiff(stacked_image, output_file)
