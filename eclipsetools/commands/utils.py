import click
import matplotlib.pyplot as plt

from eclipsetools.utils.circle_finder import find_circle
from eclipsetools.utils.image_reader import open_image


@click.command()
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
