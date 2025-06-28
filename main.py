import os

import click
import cv2
import joblib
import numpy as np
import tifffile
from joblib import Parallel
from tqdm import tqdm

from eclipsetools.alignment import find_transform
from eclipsetools.preprocessing import preprocess_for_alignment
from eclipsetools.utils.image_reader import open_image


@click.group()
def main():
    pass


def align_single_image(reference_image, image_path, low_pass_sigma, output_dir):
    """
    Process a single image alignment operation.
    :param output_dir: Directory to save the output images
    :param reference_image: Preprocessed reference image data
    :param image_path: Path to the image file to align
    :param low_pass_sigma: Standard deviation for Gaussian low-pass filter in frequency domain applied to the phase correlation.
    :return: Tuple of image path and translation vector (dy, dx)
    """
    raw_image = open_image(image_path)
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

    # Save the aligned image as a TIFF file
    orig_filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = os.path.join(output_dir, f"{orig_filename_without_ext}_aligned.tiff")
    save_tiff(aligned_image, output_filename)

    return image_path, float(scale), float(rotation_degrees), (float(translation_y), float(translation_x))


@main.command()
@click.argument('reference_image', type=click.Path(exists=True))
@click.argument('images_to_align', nargs=-1, required=True)
@click.option('--output-dir', default='output', type=click.Path(exists=False),
              help='Directory to save preprocessed images.')
@click.option('--n-jobs', default=-1, type=int, help='Number of parallel jobs. Default is -1 (all CPUs).')
@click.option('--low-pass-sigma',
              default=0.03,
              type=float,
              help='Standard deviation for Gaussian low-pass filter in frequency domain applied to the phase correlation.')
def align(reference_image, images_to_align, output_dir, n_jobs, low_pass_sigma):
    """
    Align multiple eclipse images to reference image.
    """
    ref_image = preprocess_for_alignment(open_image(reference_image))

    click.echo(f"Processing {len(images_to_align)} images...")

    output_dir_abs = os.path.abspath(output_dir)
    click.echo(f'Writing preprocessed images to directory: {output_dir_abs}')

    # Ensure the output directory exists
    os.makedirs(output_dir_abs, exist_ok=True)

    # Process all images in parallel using joblib
    results = list(tqdm(total=len(images_to_align),
                        iterable=joblib.Parallel(n_jobs=n_jobs, prefer='threads', return_as='generator')(
                            joblib.delayed(align_single_image)(ref_image, image_path, low_pass_sigma, output_dir_abs)
                            for image_path in images_to_align
                        )))

    # Print results
    for image_path, scale, rotation_degrees, (translation_y, translation_x) in results:
        click.echo(f"Image: {image_path}")
        click.echo(f"  Scale: {scale:.4f}")
        click.echo(f"  Rotation: {rotation_degrees:.4f} degrees")
        click.echo(f"  Translation: {translation_x:.4f}, {translation_y:.4f}")


def save_tiff(image: np.ndarray, output_path: str):
    """
    Save image to the specified output path as tiff
    :param image: Image to save
    :param output_path: Path where the image will be saved
    :return: None
    """
    tifffile.imwrite(output_path, image, compression='zlib')


def open_and_preprocess(image_path: str, output_dir: str):
    rgb_image = open_image(image_path)
    image_preproc = preprocess_for_alignment(rgb_image)

    orig_filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = os.path.join(output_dir, f"{orig_filename_without_ext}_preproc.tiff")
    save_tiff(image_preproc, output_filename)


@main.command()
@click.argument('images_to_preprocess', nargs=-1, required=True)
@click.option('--n-jobs', default=-1, type=int, help='Number of parallel jobs. Default is -1 (all CPUs).')
@click.option('--output-dir', default='output', type=click.Path(exists=False),
              help='Directory to save preprocessed images.')
def preprocess_only(images_to_preprocess: tuple[str], n_jobs: int, output_dir: str):
    """
    Preprocess images for alignment. The output will be 32-bit grayscale TIFF images, with both negative and positive values.
    :param images_to_preprocess: Image paths to preprocess
    :param n_jobs: Number of parallel jobs to use for preprocessing. Default is -1 (use all available CPUs).
    :param output_dir: Directory to save preprocessed images.
    """

    click.echo(f"Preprocessing {len(images_to_preprocess)} images...")

    output_dir_abs = os.path.abspath(output_dir)
    click.echo('Writing preprocessed images to directory: ' + output_dir_abs)

    # Ensure the output directory exists
    os.makedirs(output_dir_abs, exist_ok=True)

    list(
        tqdm(total=len(images_to_preprocess),
             iterable=Parallel(n_jobs=n_jobs, prefer='threads', return_as='generator_unordered')(
                 joblib.delayed(open_and_preprocess)(path, output_dir_abs) for path in images_to_preprocess)))


if __name__ == '__main__':
    main()
