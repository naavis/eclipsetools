import warnings

import click
import cv2
import numpy as np
import skimage.color
from numba_progress import ProgressBar

from eclipsetools.common.circle_finder import (
    find_circle,
    get_binary_moon_mask,
    DetectedCircle,
)
from eclipsetools.common.image_reader import open_image
from eclipsetools.common.image_writer import save_tiff
from eclipsetools.filtering import get_kernel_size, inpaint_pixels, partial_convolution


@click.group("filter")
def filter_group():
    """Commands for filtering images."""
    pass


@filter_group.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--sigma",
    type=float,
    help="Sigma for the Gaussian convolution kernel. Will be ignored if sigma_tangent and sigma_radial are provided.",
)
@click.option(
    "--sigma-tangent",
    type=float,
    help="Sigma for the tangential component of the Gaussian convolution kernel. Must be used together with sigma_radial.",
)
@click.option(
    "--sigma-radial",
    type=float,
    help="Sigma for the radial component of the Gaussian convolution kernel. Must be used together with sigma_tangent.",
)
@click.option(
    "--filter-amount",
    type=float,
    help="Amount of unsharp masking to apply.",
    required=True,
)
@click.option("--mask-path", type=click.Path(exists=True), help="Path to mask image.")
@click.option(
    "--output-file",
    type=click.Path(),
    default="unsharp_masked_image.tiff",
    help="Output filename for the filtered image tiff file.",
)
@click.option(
    "--moon-min-radius",
    type=int,
    default=200,
    help="Minimum radius of the moon in pixels for moon detection.",
)
@click.option(
    "--moon-max-radius",
    type=int,
    default=2000,
    help="Maximum radius of the moon in pixels for moon detection.",
)
def unsharp_mask(
    input_file: str,
    sigma: float,
    sigma_tangent: float,
    sigma_radial: float,
    filter_amount: float,
    output_file: str,
    mask_path: str,
    moon_min_radius: int,
    moon_max_radius: int,
):
    """
    Process image with an adaptive unsharp mask filter.
    The filter uses partial convolution and a spatially varying convolution kernel.
    """

    validate_sigma_parameters(sigma, sigma_tangent, sigma_radial)
    if sigma is not None:
        sigma_tangent = sigma_radial = sigma

    image = open_image(input_file)
    lab_image = skimage.color.rgb2lab(image)
    image_l = lab_image[:, :, 0] / 100.0  # Scale L channel to [0, 1]

    moon_params = find_circle(image_l, moon_min_radius, moon_max_radius)
    if mask_path:
        click.echo(f"Using mask from {mask_path}")
        moon_mask = open_image(mask_path) == 1.0
    else:
        click.echo("Finding moon in the image")
        moon_mask = get_binary_moon_mask(image_l.shape, moon_params, 1.005)

    convolved_image = convolve_with_infill(
        image_l, moon_mask, moon_params, sigma_radial, sigma_tangent
    )

    filtered_image = np.clip(
        image_l + filter_amount * (image_l - convolved_image), 0.0, 1.0
    )

    # Not all CIELAB value combinations are valid, and the conversion to RGB complains, but we ignore that.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Conversion from CIE-LAB")
        processed_rgb = skimage.color.lab2rgb(
            np.stack(
                [filtered_image * 100, lab_image[:, :, 1], lab_image[:, :, 2]], axis=-1
            )
        )

    click.echo(f"Saving filtered image to {output_file}")
    save_tiff(processed_rgb, output_file, embed_srgb=True)


@filter_group.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--filter-params",
    "-f",
    type=(float, float),
    multiple=True,
    help='Convolution kernel sigma and filter amount pairs, e.g. "1.0 100.0"',
)
@click.option("--mask-path", type=click.Path(exists=True), help="Path to mask image.")
@click.option(
    "--output-file",
    type=click.Path(),
    default="achf_image.tiff",
    help="Output filename for the filtered image tiff file.",
)
@click.option(
    "--moon-min-radius",
    type=int,
    default=200,
    help="Minimum radius of the moon in pixels for moon detection.",
)
@click.option(
    "--moon-max-radius",
    type=int,
    default=2000,
    help="Maximum radius of the moon in pixels for moon detection.",
)
def achf(
    input_file: str,
    filter_params: tuple[tuple[float, float], ...] | None,
    output_file: str,
    mask_path: str,
    moon_min_radius: int,
    moon_max_radius: int,
):
    """
    Apply the Adaptive Circular High-Frequency filter (ACHF) to the image.
    The filter combines several unsharp masks with different parameters to enhance the image,
    using partial convolution to avoid ringing artifacts around the moon.
    """
    if not filter_params:
        filter_params = [
            (1.0, 100.0),
            (2.0, 80.0),
            (4.0, 60.0),
            (8.0, 30.0),
            (16.0, 10.0),
        ]

    image = open_image(input_file)
    lab_image = skimage.color.rgb2lab(image)
    image_l = lab_image[:, :, 0] / 100.0  # Scale L channel to [0, 1]

    click.echo(
        f"Finding moon in the image with radius range {moon_min_radius} to {moon_max_radius} px"
    )
    moon_params = find_circle(image_l, moon_min_radius, moon_max_radius)
    click.echo(
        f"Moon center x = {moon_params.center[1]:.2f}, y = {moon_params.center[0]:.2f}, radius: {moon_params.radius:.2f}"
    )
    if mask_path:
        click.echo(f"Using mask from {mask_path}")
        moon_mask = open_image(mask_path) == 1.0
    else:
        click.echo("Finding moon in the image")
        moon_mask = get_binary_moon_mask(image_l.shape, moon_params, 1.005)

    filtered_image = image_l.copy()

    for filter_radius, filter_amount in filter_params:
        click.echo(
            f"Applying filter with radius {filter_radius} and amount {filter_amount}"
        )
        filtered_image += filter_amount * (
            image_l
            - convolve_with_infill(
                image_l, moon_mask, moon_params, filter_radius, filter_radius
            )
        )

    # Not all CIELAB value combinations are valid, and the conversion to RGB complains, but we ignore that.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Conversion from CIE-LAB")
        processed_rgb = skimage.color.lab2rgb(
            np.stack(
                [
                    np.clip(filtered_image, 0.0, 1.0) * 100,
                    lab_image[:, :, 1],
                    lab_image[:, :, 2],
                ],
                axis=-1,
            )
        )

    click.echo(f"Saving filtered image to {output_file}")
    save_tiff(processed_rgb, output_file, embed_srgb=True)


def convolve_with_infill(
    image: np.ndarray,
    mask: np.ndarray,
    moon_params: DetectedCircle,
    sigma_radial: float,
    sigma_tangent: float,
) -> np.ndarray:
    """
    Convolve the image with a partial convolution kernel, infilling masked pixels to avoid ringing artifacts.
    :param image: 2D array of the image to be filtered.
    :param mask: 2D boolean array where True indicates pixels to be convolved.
    :param moon_params: Parameters of the detected moon circle, used for partial convolution.
    :param sigma_radial: Sigma for the radial component of the Gaussian convolution kernel.
    :param sigma_tangent: Sigma for the tangential component of the Gaussian convolution kernel.
    :return: Convolved image with the same shape as the input image.
    """
    kernel_size = get_kernel_size(sigma_tangent, sigma_radial)

    dilated_mask = cv2.dilate(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
    ).astype(np.bool)
    pixels_to_infill = np.logical_xor(mask, dilated_mask)

    with ProgressBar(
        total=image.shape[0], unit="row", desc="Inpainting pixels"
    ) as progress_proxy:
        inpainted_image = inpaint_pixels(
            image, pixels_to_infill, mask, kernel_size, progress_proxy
        )

    if sigma_radial == sigma_tangent:
        convolved_image = cv2.GaussianBlur(
            inpainted_image,
            (kernel_size, kernel_size),
            sigma_tangent,
        )
    else:
        convolved_image = partial_convolution(
            inpainted_image,
            np.ones_like(mask),
            sigma_tangent,
            sigma_radial,
            moon_params.center,
        )

    convolved_image = np.where(mask, convolved_image, image)
    return convolved_image


def validate_sigma_parameters(
    sigma: float, sigma_tangent: float, sigma_radial: float
) -> None:
    """
    Validate the sigma parameters for the unsharp mask command.
    You must provide either a single --sigma value or both --sigma-tangent and --sigma-radial.

    :raises click.BadParameter: If the parameters are not valid.
    """
    s = sigma is not None
    st = sigma_tangent is not None
    sr = sigma_radial is not None
    if (s and not (st or sr)) or (not s and (st and sr)):
        return

    raise click.BadParameter(
        "You must provide either a single sigma value or both --sigma-tangent and --sigma-radial"
    )
