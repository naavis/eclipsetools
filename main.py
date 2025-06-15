import glob
import os

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
    image_to_align = preprocess_for_alignment(open_raw_image(image_path))
    translation = find_translation(reference_image, image_to_align)
    return image_path, translation


@click.command()
@click.argument('reference_image', type=click.Path(exists=True))
@click.argument('images_to_align', nargs=-1, required=True)
@click.option('--n-jobs', default=-1, type=int, help='Number of parallel jobs. Default is -1 (all CPUs).')
def main(reference_image, images_to_align, n_jobs):
    """
    Align multiple eclipse images based on translation.

    REFERENCE_IMAGE: Path to the reference RAW image file
    IMAGES_TO_ALIGN: One or more RAW image files or glob patterns that need alignment
    """
    # Expand glob patterns to get actual file paths
    expanded_image_paths = []
    for path_pattern in images_to_align:
        # If this is a direct file path
        if os.path.exists(path_pattern):
            expanded_image_paths.append(path_pattern)
        else:
            # Treat as a glob pattern
            matching_files = glob.glob(path_pattern)
            if matching_files:
                expanded_image_paths.extend(matching_files)
            else:
                click.echo(f"Warning: Pattern '{path_pattern}' did not match any files", err=True)

    if not expanded_image_paths:
        click.echo("Error: No valid image files found to align", err=True)
        return

    ref_image = preprocess_for_alignment(open_raw_image(reference_image))

    click.echo(f"Processing {len(expanded_image_paths)} images...")

    # Process all images in parallel using joblib
    results = joblib.Parallel(n_jobs=n_jobs, prefer='threads')(
        joblib.delayed(align_single_image)(ref_image, image_path)
        for image_path in expanded_image_paths
    )

    # Print results
    for image_path, translation in results:
        click.echo(f"Image: {image_path}")
        click.echo(f"  Translation: {translation}")


if __name__ == '__main__':
    main()
