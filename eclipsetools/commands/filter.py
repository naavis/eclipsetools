import warnings

import click
import cv2
import numpy as np
import skimage.color
from numba_progress import ProgressBar

from eclipsetools.common.circle_finder import find_circle, get_binary_moon_mask
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

    kernel_size = get_kernel_size(sigma_tangent, sigma_radial)
    dilated_mask = cv2.dilate(
        moon_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
    ).astype(np.bool)

    pixels_to_infill = np.logical_xor(moon_mask, dilated_mask)

    with ProgressBar(
        total=image.shape[0], unit="row", desc="Inpainting pixels"
    ) as progress_proxy:
        inpainted_image = inpaint_pixels(
            image_l, pixels_to_infill, moon_mask, kernel_size, progress_proxy
        )

    convolved_image = partial_convolution(
        inpainted_image,
        np.ones_like(moon_mask),
        sigma_tangent,
        sigma_radial,
        moon_params.center,
    )
    convolved_image = np.where(moon_mask, convolved_image, image_l)

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


def validate_sigma_parameters(sigma, sigma_tangent, sigma_radial):
    s = sigma is not None
    st = sigma_tangent is not None
    sr = sigma_radial is not None
    if (s and not (st or sr)) or (not s and (st and sr)):
        return

    raise click.BadParameter(
        "You must provide either a single sigma value or both --sigma-tangent and --sigma-radial"
    )
