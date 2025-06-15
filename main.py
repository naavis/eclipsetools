import glob
import os

import click
import joblib

from eclipsetools.alignment import find_translation
from eclipsetools.preprocessing import preprocess_for_alignment
from eclipsetools.utils.raw_reader import open_raw_image


def align_single_image(reference_image, image_path, low_pass_sigma):
    """
    Process a single image alignment operation.
    :param reference_image: Preprocessed reference image data
    :param image_path: Path to the image file to align
    :param low_pass_sigma: Standard deviation for Gaussian low-pass filter in frequency domain applied to the phase correlation.
    :return: Tuple of image path and translation vector (dy, dx)
    """
    image_to_align = preprocess_for_alignment(open_raw_image(image_path))
    translation = find_translation(reference_image, image_to_align, low_pass_sigma)
    return image_path, translation


@click.command()
@click.argument('reference_image', type=click.Path(exists=True))
@click.argument('images_to_align', nargs=-1, required=True)
@click.option('--n-jobs', default=-1, type=int, help='Number of parallel jobs. Default is -1 (all CPUs).')
@click.option('--low-pass-sigma',
              default=0.115,
              type=float,
              help='Standard deviation for Gaussian low-pass filter in frequency domain applied to the phase correlation.')
def main(reference_image, images_to_align, n_jobs, low_pass_sigma):
    """
    Align multiple eclipse images based on translation.
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
        joblib.delayed(align_single_image)(ref_image, image_path, low_pass_sigma)
        for image_path in expanded_image_paths
    )

    # Print results
    for image_path, translation in results:
        click.echo(f"Image: {image_path}")
        click.echo(f"  Translation: {translation}")


if __name__ == '__main__':
    main()
