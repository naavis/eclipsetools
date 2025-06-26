import glob
import os

import click
import cv2
import joblib
import numpy as np

from eclipsetools.alignment import find_transform
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
    raw_image = open_raw_image(image_path)
    image_to_align = preprocess_for_alignment(raw_image)
    scale, rotation_degrees, (translation_y, translation_x) = find_transform(
        reference_image,
        image_to_align,
        low_pass_sigma,
        allow_scale=False)

    rotation_scale_matrix = np.vstack([
        cv2.getRotationMatrix2D((image_to_align.shape[1] / 2, image_to_align.shape[0] / 2),
                                -rotation_degrees,
                                1.0 / scale),
        [0.0, 0.0, 1.0]], dtype=np.float32)
    translation_matrix = np.array(
        [[1, 0, -translation_x],
         [0, 1, -translation_y],
         [0, 0, 1]], dtype=np.float32)

    transform_matrix = (rotation_scale_matrix @ translation_matrix)[:2, :]
    aligned_image = cv2.warpAffine(raw_image, transform_matrix, (image_to_align.shape[1], image_to_align.shape[0]),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    name_prefix = os.path.splitext(os.path.basename(image_path))[0]
    save_output(output_dir, name_prefix, aligned_image, image_to_align)
    return image_path, float(scale), float(rotation_degrees), (float(translation_y), float(translation_x))


def save_output(output_dir, name_prefix, aligned_image, preprocessed_image):
    aligned_image_path = os.path.join(output_dir, f"{name_prefix}_aligned.tiff")
    cv2.imwrite(aligned_image_path, (aligned_image * (2 ** 16 - 1)).astype(np.uint16))

    cv2.imwrite(os.path.join(output_dir, f"{name_prefix}_preprocessed.tiff"),
                (((preprocessed_image - np.min(preprocessed_image)) / (
                        np.max(preprocessed_image) - np.min(preprocessed_image))) * 255).astype(np.uint8))


@click.command()
@click.argument('reference_image', type=click.Path(exists=True))
@click.argument('images_to_align', nargs=-1, required=True)
@click.option('--n-jobs', default=-1, type=int, help='Number of parallel jobs. Default is -1 (all CPUs).')
@click.option('--low-pass-sigma',
              default=0.03,
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
    for image_path, scale, rotation_degrees, (translation_y, translation_x) in results:
        click.echo(f"Image: {image_path}")
        click.echo(f"  Scale: {scale:.4f}")
        click.echo(f"  Rotation: {rotation_degrees:.4f} degrees")
        click.echo(f"  Translation: {translation_x:.4f}, {translation_y:.4f}")


if __name__ == '__main__':
    main()
