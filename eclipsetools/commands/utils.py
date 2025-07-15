import click

from eclipsetools.utils.circle_finder import find_circle
from eclipsetools.utils.image_reader import open_image


@click.command()
@click.argument(
    "image_path", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option("--min-radius", default=400, help="Minimum radius of the moon in pixels.")
@click.option("--max-radius", default=600, help="Maximum radius of the moon in pixels.")
def find_moon(image_path: str, min_radius: int, max_radius: int):
    """
    Find the moon in an image.
    """
    image = open_image(image_path).mean(axis=2)
    circle = find_circle(image, min_radius=min_radius, max_radius=max_radius)
    click.echo(
        f"Found moon at x: {circle.center[1]:.2f}, y: {circle.center[0]:.2f}, with radius: {circle.radius:.2f} pixels"
    )
