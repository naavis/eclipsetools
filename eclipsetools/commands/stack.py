from pathlib import Path

import click
import joblib
import numpy as np
from tqdm import tqdm

from eclipsetools.common.circle_finder import find_circle, get_binary_moon_mask
from eclipsetools.common.image_reader import open_image
from eclipsetools.common.image_writer import save_tiff
from eclipsetools.stacking.linear_fit import fit_eclipse_image_pair
from eclipsetools.stacking.linear_fit import solve_global_linear_fits
from eclipsetools.stacking.sorting import sort_images_by_brightness
from eclipsetools.stacking.weighting import weight_function_hat


@click.group("stack")
def stack_group():
    """
    Image stacking commands for eclipse images.
    :return:
    """
    pass


@stack_group.command()
@click.argument("images_to_stack", nargs=-1, required=True)
@click.option(
    "--output-file",
    type=click.Path(),
    default="average_stacked_image.tiff",
    help="Output filename for the stacked image tiff file.",
)
def average(images_to_stack: list[str], output_file: str):
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


@stack_group.command()
@click.argument("reference_image", type=click.Path(exists=True, dir_okay=False))
@click.argument("images_to_stack", nargs=-1, required=True)
@click.option(
    "--n-jobs",
    default=-1,
    type=int,
    help="Number of parallel jobs. Default is -1 (all CPUs).",
)
@click.option(
    "--fit-intercept",
    is_flag=True,
    default=False,
    help="Fit the intercept in the linear regression. If not set, the linear fit is forced through the origin."
    "This option might improve fit quality for uncalibrated images.",
)
@click.option(
    "--output-file",
    type=click.Path(),
    default="hdr_stacked_image.tiff",
    help="Output filename for the stacked image tiff file.",
)
@click.option(
    "--moon-min-radius",
    type=int,
    default=200,
    help="Minimum radius of the moon in pixels for circle detection.",
)
@click.option(
    "--moon-max-radius",
    type=int,
    default=2000,
    help="Maximum radius of the moon in pixels for circle detection.",
)
def hdr(
    reference_image: str,
    images_to_stack: list[str],
    fit_intercept: bool,
    n_jobs: int,
    output_file: str,
    moon_min_radius: int,
    moon_max_radius: int,
):
    """
    Stack multiple images to HDR stack. Images must be pre-aligned. Images taken with different exposure times
    are combined by linear fitting to the reference image. The output is a 64-bit floating point TIFF image.

    The reference image must be one of the images in the list of images to stack.
    """

    reference_image = str(Path(reference_image).resolve())
    images_to_stack = [str(Path(img).resolve()) for img in images_to_stack]

    if reference_image not in images_to_stack:
        click.echo(
            f"Reference image {reference_image} is not in the list of images to stack.",
            err=True,
        )
        return

    image_pairs = form_image_pairs(images_to_stack, n_jobs)

    # Size of the moon mask relative to the moon radius for linear fitting
    linear_fit_mask_size = 1.01

    # Linear fit each pair of images
    linear_fit_results = list(
        tqdm(
            iterable=joblib.Parallel(
                n_jobs=n_jobs, prefer="threads", return_as="generator"
            )(
                joblib.delayed(fit_eclipse_image_pair)(
                    image_a, image_b, fit_intercept, linear_fit_mask_size
                )
                for image_a, image_b in image_pairs
            ),
            total=len(image_pairs),
            desc="Pairwise linear fitting",
            unit="pair",
        )
    )

    click.echo("Solving global linear fit for all images")
    linear_fits = solve_global_linear_fits(linear_fit_results, reference_image)

    ref_image_data = open_image(reference_image)

    ref_moon_params = find_circle(
        ref_image_data[:, :, 1], moon_min_radius, moon_max_radius
    )
    # We use a smaller mask for the reference image to have a cleaner moon edge in the final stack
    ref_moon_weighting_mask_size = 1.005
    ref_moon_mask = get_binary_moon_mask(
        ref_image_data.shape, ref_moon_params, ref_moon_weighting_mask_size
    )

    total_weights = np.zeros_like(ref_image_data, dtype=np.float64)
    weighted_sum = np.zeros_like(ref_image_data, dtype=np.float64)

    # Size of the moon mask relative to the moon radius for stacking
    weighting_mask_size = 1.01

    # Process images in parallel for stacking
    for image in tqdm(
        iterable=joblib.Parallel(
            n_jobs=n_jobs, prefer="threads", return_as="generator_unordered"
        )(
            joblib.delayed(_process_image_for_stacking)(
                image_path,
                linear_fits[image_path],
                ref_moon_mask,
                weighting_mask_size,
                ref_moon_params.radius,
                moon_min_radius,
                moon_max_radius,
            )
            for image_path in linear_fits.keys()
        ),
        total=len(linear_fits),
        desc="Compositing images",
        unit="img",
    ):
        weights, calibrated_image = image
        weighted_sum += weights * calibrated_image
        total_weights += weights

    # Masked and saturated pixels may end up being unfilled in the final image.
    # These pixels are filled with data from the reference image.
    unfilled_pixels = total_weights == 0
    total_weights[unfilled_pixels] = 1.0  # Avoid division by zero
    stacked_image = weighted_sum / total_weights
    stacked_image[unfilled_pixels] = ref_image_data[unfilled_pixels]

    # Ensure there are no negative values, and normalize values to maximum of 1.0
    stacked_image -= min(stacked_image.min(), 0.0)
    stacked_image /= max(stacked_image.max(), 1.0)
    click.echo(f"Saving stacked image to {output_file}")
    save_tiff(stacked_image, output_file)


def form_image_pairs(images_to_stack, n_jobs, show_progress=True):
    # Sort images by brightness
    sorted_images = [
        path
        for path, _brightness in sort_images_by_brightness(
            images_to_stack, n_jobs, show_progress=show_progress
        )
    ]

    # Form pairs of consecutive images for linear fitting
    image_pairs = [
        (sorted_images[i], sorted_images[i + 1]) for i in range(len(sorted_images) - 1)
    ]
    return image_pairs


def _process_image_for_stacking(
    image_path: str,
    linear_fit_params: tuple[float, float],
    ref_moon_mask: np.ndarray,
    mask_size: float,
    ref_moon_size_px: float,
    min_moon_radius: int,
    max_moon_radius: int,
):
    linear_coef, linear_intercept = linear_fit_params
    image = open_image(image_path)
    moon_params = find_circle(image[:, :, 1], min_moon_radius, max_moon_radius)
    # The moon radius can be underestimated in very saturated images, so we use the reference moon size
    moon_params.radius = ref_moon_size_px

    weights = weight_function_hat(image)
    image_moon_mask = get_binary_moon_mask(image.shape, moon_params, mask_size)
    pixels_with_moon = ~ref_moon_mask | ~image_moon_mask
    weights[pixels_with_moon] = 0.0
    calibrated_image = (image - linear_intercept) / linear_coef
    return weights, calibrated_image
