import click

from eclipsetools.alignment import find_translation
from eclipsetools.preprocessing import preprocess_for_alignment
from eclipsetools.utils.raw_reader import open_raw_image


@click.command()
@click.argument('reference_image', type=click.Path(exists=True))
@click.argument('image_to_align', type=click.Path(exists=True))
def main(reference_image, image_to_align):
    """
    Align two eclipse images based on translation.

    REFERENCE_IMAGE: Path to the reference RAW image file
    IMAGE_TO_ALIGN: Path to the RAW image file that needs alignment
    """
    ref_image = preprocess_for_alignment(open_raw_image(reference_image).data)
    img_to_align = preprocess_for_alignment(open_raw_image(image_to_align).data)
    print(find_translation(ref_image, img_to_align))


if __name__ == '__main__':
    main()
