import click
import matplotlib.pyplot as plt
import numpy as np

from eclipsetools.common.circle_finder import find_circle
from eclipsetools.common.image_reader import open_image
from eclipsetools.common.image_writer import save_tiff
from eclipsetools.common.moon_masker import get_precise_moon_mask


@click.group("utils")
def utils_group():
    """
    Utility commands for image processing.
    """
    pass


@utils_group.command()
@click.argument(
    "image_path", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option("--min-radius", default=400, help="Minimum radius of the moon in pixels.")
@click.option("--max-radius", default=600, help="Maximum radius of the moon in pixels.")
@click.option(
    "--plot-circle", is_flag=True, help="Plot the detected circle on the image."
)
def find_moon(image_path: str, min_radius: int, max_radius: int, plot_circle: bool):
    """
    Find the moon in an image.
    """
    image = open_image(image_path).mean(axis=2)
    circle = find_circle(image, min_radius=min_radius, max_radius=max_radius)
    click.echo(
        f"Found moon at x: {circle.center[1]:.2f}, y: {circle.center[0]:.2f}, with radius: {circle.radius:.2f} pixels"
    )
    if plot_circle:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray")
        circle_patch = plt.Circle(
            (circle.center[1], circle.center[0]), circle.radius, color="red", fill=False
        )
        ax.add_patch(circle_patch)
        ax.set_title("Detected Moon Circle")
        plt.axis("off")
        plt.show()


@utils_group.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option(
    "--min-moon-radius",
    type=int,
    default=400,
    help="Minimum radius of the moon in pixels for mask creation.",
)
@click.option(
    "--max-moon-radius",
    type=int,
    default=2000,
    help="Maximum radius of the moon in pixels for mask creation.",
)
def create_moon_mask(
    image_path: str, output_file: str, min_moon_radius: int, max_moon_radius: int
):
    """
    Create a precise moon mask from an image. The sky will have the value 1.0 and the moon 0.0.
    This command should only be used with linear images.
    """
    image = open_image(image_path)

    moon_mask = get_precise_moon_mask(image, min_moon_radius, max_moon_radius)

    click.echo(f"Saving mask to {output_file}")
    save_tiff(moon_mask, output_file)


@utils_group.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.argument("amount", type=float)
def log_stretch(input_file: str, output_file: str, amount: float):
    """
    Apply logarithmic stretch to an image.
    The stretch defined by log(amount * image + 1) / log(amount + 1).
    """
    image = open_image(input_file)
    stretched_image = np.log1p(amount * image) / np.log1p(amount)
    stretched_image = np.clip(stretched_image, 0.0, 1.0)

    click.echo(f"Saving stretched image to {output_file}")
    save_tiff(stretched_image, output_file)


@utils_group.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
def color_calibrate(input_file: str, output_file: str):
    """
    Color calibrate image.
    The calibration is done by assuming the image is a linear image, and the background and solar corona are neutral.
    The background value is subtracted from the image, and the color channels are divided by suitable ratios to make
    the solar corona neutral while keeping the background neutral as well.
    """
    image = open_image(input_file)
    assert image.ndim == 3, "Image must be a 3D array (height, width, channels)."

    small_image = image[::10, ::10, :]  # Downsample for performance

    valid_pixels = (
        (small_image[:, :, 0] > 0.0)
        & (small_image[:, :, 0] < 1.0)
        & (small_image[:, :, 1] > 0.0)
        & (small_image[:, :, 1] < 1.0)
        & (small_image[:, :, 2] > 0.0)
        & (small_image[:, :, 2] < 1.0)
    )
    red = small_image[:, :, 0]
    red = red[valid_pixels]

    green = small_image[:, :, 1]
    green = green[valid_pixels]

    blue = small_image[:, :, 2]
    blue = blue[valid_pixels]

    r_bg = np.percentile(red, 1)
    g_bg = np.percentile(green, 1)
    b_bg = np.percentile(blue, 1)

    r_fg = np.percentile(red, 99)
    g_fg = np.percentile(green, 99)
    b_fg = np.percentile(blue, 99)

    rg_ratio = (r_fg - r_bg) / (g_fg - g_bg)
    bg_ratio = (b_fg - b_bg) / (g_fg - g_bg)

    # Ensure all ratios are below 1.0 to avoid non-white saturated pixels
    ratios = np.array([rg_ratio, 1.0, bg_ratio])
    ratios /= ratios.max()

    click.echo(f"Background RGB: ({r_bg:.4f}, {g_bg:.4f}, {b_bg:.4f})")
    click.echo(f"White balance ratios: {ratios}")

    # Subtract the background and divide by the ratios
    image_cc = (image - np.array([r_bg, g_bg, b_bg])) / ratios

    # Ensure no negative values, and don't clip more pixels than necessary
    min_value = min(image_cc.min(), 0.0)
    image_cc = (image_cc - min_value) / (1.0 - min_value)
    image_cc = np.clip(image_cc, 0.0, 1.0)

    click.echo(f"Saving color-calibrated image to {output_file}")
    save_tiff(image_cc, output_file)
