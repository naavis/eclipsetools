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
from eclipsetools.stacking import linear_fit, weight_function_sigmoid
from eclipsetools.utils.circle_finder import find_circle
from eclipsetools.utils.image_reader import open_image


@click.group()
def main():
    pass


def align_single_image(reference_image: np.ndarray,
                       image_path: str,
                       low_pass_sigma: float,
                       output_dir: str,
                       mask_inner_radius: float,
                       mask_outer_radius: float) -> tuple[str, float, float, tuple[float, float]]:
    """
    Process a single image alignment operation.
    :param output_dir: Directory to save the output images
    :param reference_image: Preprocessed reference image data
    :param image_path: Path to the image file to align
    :param low_pass_sigma: Standard deviation for Gaussian low-pass filter in frequency domain applied to the phase correlation.
    :param mask_inner_radius: Inner radius of the annulus mask in multiples of the moon radius.
    :param mask_outer_radius: Outer radius of the annulus mask in multiples of the inner radius. Set to -1 to only mask the moon.
    :return: Tuple of image path and translation vector (dy, dx)
    """
    raw_image = open_image(image_path)
    image_to_align = preprocess_for_alignment(raw_image, mask_inner_radius, mask_outer_radius)
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
                                   flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

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
@click.option('--mask-inner-radius', default=1.2, type=float, help='Inner radius of the annulus mask in multiples of '
                                                                   'the moon radius.')
@click.option('--mask-outer-radius', default=2.0, type=float, help='Outer radius of the annulus mask in multiples of '
                                                                   'the inner radius. Set to -1 to only mask the moon.')
def align(reference_image, images_to_align, output_dir, n_jobs, low_pass_sigma, mask_inner_radius, mask_outer_radius):
    """
    Align multiple eclipse images to reference image.
    """
    ref_image = preprocess_for_alignment(open_image(reference_image), mask_inner_radius, mask_outer_radius)

    click.echo(f"Processing {len(images_to_align)} images...")

    output_dir_abs = os.path.abspath(output_dir)
    click.echo(f'Writing aligned images to directory: {output_dir_abs}')

    # Ensure the output directory exists
    os.makedirs(output_dir_abs, exist_ok=True)

    # Process all images in parallel using joblib
    results = list(tqdm(total=len(images_to_align),
                        iterable=joblib.Parallel(n_jobs=n_jobs, prefer='threads', return_as='generator_unordered')(
                            joblib.delayed(align_single_image)(ref_image, image_path, low_pass_sigma, output_dir_abs,
                                                               mask_inner_radius, mask_outer_radius)
                            for image_path in images_to_align
                        )))

    results.sort(key=lambda x: x[0])  # Sort results by image path

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


def open_and_preprocess(image_path: str, output_dir: str, mask_inner_radius: float, mask_outer_radius: float):
    rgb_image = open_image(image_path)
    image_preproc = preprocess_for_alignment(rgb_image, mask_inner_radius, mask_outer_radius)

    # Normalize the image to have mean 0 and std 1, then shift to have mean 0.5, so it is easier to view in an external program
    image_preproc = np.clip((image_preproc - image_preproc.mean()) / image_preproc.std() + 0.5, 0.0, 1.0)

    orig_filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = os.path.join(output_dir, f"{orig_filename_without_ext}_preproc.tiff")
    save_tiff(image_preproc, output_filename)


@main.command()
@click.argument('images_to_preprocess', nargs=-1, required=True)
@click.option('--n-jobs', default=-1, type=int, help='Number of parallel jobs. Default is -1 (all CPUs).')
@click.option('--output-dir', default='output', type=click.Path(exists=False),
              help='Directory to save preprocessed images.')
@click.option('--mask-inner-radius', default=1.2, type=float, help='Inner radius of the annulus mask in multiples of '
                                                                   'the moon radius.')
@click.option('--mask-outer-radius', default=2.0, type=float, help='Outer radius of the annulus mask in multiples of '
                                                                   'the inner radius. Set to -1 to only mask the moon.')
def preprocess_only(images_to_preprocess: tuple[str], n_jobs: int, output_dir: str, mask_inner_radius: float,
                    mask_outer_radius: float):
    """
    Preprocess images for alignment. The output will be 32-bit grayscale TIFF images, with both negative and positive values.
    """

    click.echo(f"Preprocessing {len(images_to_preprocess)} images...")

    output_dir_abs = os.path.abspath(output_dir)
    click.echo('Writing preprocessed images to directory: ' + output_dir_abs)

    # Ensure the output directory exists
    os.makedirs(output_dir_abs, exist_ok=True)

    list(
        tqdm(total=len(images_to_preprocess),
             iterable=Parallel(n_jobs=n_jobs, prefer='threads', return_as='generator_unordered')(
                 joblib.delayed(open_and_preprocess)(path, output_dir_abs, mask_inner_radius, mask_outer_radius) for
                 path in
                 images_to_preprocess)))


@main.command()
@click.argument('reference_image', type=click.Path(exists=True))
@click.argument('images_to_stack', nargs=-1, required=True)
@click.option('--output-file', type=click.Path(), default='stacked_image.tiff',
              help='Output filename for the stacked image tiff file.')
def stack(reference_image: str, images_to_stack: tuple[str], output_file: str):
    ref_image = open_image(reference_image)
    ref_image -= min(ref_image.min(), 0.0)  # Ensure the reference image is non-negative

    # Mask out moving moon, because it confuses the linear fitting
    moon_params = find_circle(ref_image[:, :, 1], min_radius=400, max_radius=600)
    y, x = np.ogrid[:ref_image.shape[0], :ref_image.shape[1]]
    distances = np.sqrt((x - moon_params.center[1]) ** 2 + (y - moon_params.center[0]) ** 2)
    moon_mask = distances >= int(1.2 * moon_params.radius)

    ref_points = ref_image[moon_mask, :].ravel()

    total_weights = np.zeros_like(ref_image)
    weighted_sum = np.zeros_like(ref_image)

    for image_path in images_to_stack:
        click.echo(f"Image: {image_path}")
        image = open_image(image_path)
        assert image.shape == ref_image.shape, 'Stacked images must have the same shape'

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
    tifffile.imwrite(output_file, stacked_image, compression='zlib')


if __name__ == '__main__':
    main()
