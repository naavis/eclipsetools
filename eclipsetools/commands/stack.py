import click
import numpy as np

from eclipsetools.stacking import linear_fit, weight_function_sigmoid
from eclipsetools.utils.circle_finder import find_circle
from eclipsetools.utils.image_reader import open_image
from eclipsetools.utils.image_writer import save_tiff


@click.command()
@click.argument("reference_image", type=click.Path(exists=True))
@click.argument("images_to_stack", nargs=-1, required=True)
@click.option(
    "--output-file",
    type=click.Path(),
    default="stacked_image.tiff",
    help="Output filename for the stacked image tiff file.",
)
def stack(reference_image: str, images_to_stack: tuple[str], output_file: str):
    """
    Stack multiple eclipse images together. Images must be pre-aligned. Images taken with different exposure times
    are combined by linear fitting to the reference image.
    """
    ref_image = open_image(reference_image)
    ref_image -= min(ref_image.min(), 0.0)  # Ensure the reference image is non-negative

    # Mask out moving moon, because it confuses the linear fitting
    moon_params = find_circle(ref_image[:, :, 1], min_radius=400, max_radius=600)
    y, x = np.ogrid[: ref_image.shape[0], : ref_image.shape[1]]
    distances = np.sqrt(
        (x - moon_params.center[1]) ** 2 + (y - moon_params.center[0]) ** 2
    )
    moon_mask = distances >= int(1.2 * moon_params.radius)

    ref_points = ref_image[moon_mask, :].ravel()

    total_weights = np.zeros_like(ref_image)
    weighted_sum = np.zeros_like(ref_image)

    # TODO: Handle weights properly for the moon area, like Druckmüller describes
    for image_path in images_to_stack:
        click.echo(f"Image: {image_path}")
        image = open_image(image_path)
        assert image.shape == ref_image.shape, "Stacked images must have the same shape"

        image_points = image[moon_mask, :].ravel()
        linear_coef, linear_intercept = linear_fit(ref_points, image_points)
        click.echo(f"Linear fit: y = {linear_coef:.4f} * x + {linear_intercept:.4f}")

        weights = weight_function_sigmoid(image)
        weighted_sum += weights * (image - linear_intercept) / linear_coef
        total_weights += weights

    stacked_image = weighted_sum / total_weights

    # Ensure there are no negative values, and normalize values to maximum of 1.0
    stacked_image -= min(stacked_image.min(), 0.0)
    stacked_image /= max(stacked_image.max(), 1.0)
    click.echo(f"Saving stacked image to {output_file}")
    save_tiff(stacked_image, output_file)
