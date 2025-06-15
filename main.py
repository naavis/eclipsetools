import click
import joblib

from eclipsetools.alignment import find_translation
from eclipsetools.preprocessing import preprocess_for_alignment
from eclipsetools.utils.raw_reader import open_raw_image


def align_single_image(reference_image, image_path):
    """Process a single image alignment operation.

    Args:
        reference_image: Preprocessed reference image data
        image_path: Path to the image file to align

    Returns:
        tuple: (image_path, translation_result)
    """
    image_to_align = preprocess_for_alignment(open_raw_image(image_path).data)
    translation = find_translation(reference_image, image_to_align)
    return image_path, translation


@click.command()
@click.argument('reference_image', type=click.Path(exists=True))
@click.argument('images_to_align', type=click.Path(exists=True), nargs=-1, required=True)
@click.option('--n-jobs', default=-1, type=int, help='Number of parallel jobs. Default is -1 (all CPUs).')
def main(reference_image, images_to_align, n_jobs):
    """
    Align multiple eclipse images based on translation.

    REFERENCE_IMAGE: Path to the reference RAW image file
    IMAGES_TO_ALIGN: One or more RAW image files that need alignment
    """
    ref_image = preprocess_for_alignment(open_raw_image(reference_image).data)

    print(f"Processing {len(images_to_align)} images in parallel...")

    # Process all images in parallel using joblib
    results = joblib.Parallel(n_jobs=n_jobs, prefer='threads')(
        joblib.delayed(align_single_image)(ref_image, image_path)
        for image_path in images_to_align
    )

    # Print results
    for image_path, translation in results:
        print(f"Image: {image_path}")
        print(f"  Translation: {translation}")


if __name__ == '__main__':
    main()
