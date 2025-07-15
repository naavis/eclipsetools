import os

import click
import cv2
import joblib
import numpy as np
from tqdm import tqdm

from eclipsetools.alignment import find_transform
from eclipsetools.preprocessing import preprocess_with_auto_mask
from eclipsetools.preprocessing.masking import MaskMode
from eclipsetools.utils.image_reader import open_image
from eclipsetools.utils.image_writer import save_tiff


@click.command()
@click.argument('reference_image', type=click.Path(exists=True))
@click.argument('images_to_align', nargs=-1, required=True)
@click.option('--output-dir',
              default='output',
              type=click.Path(exists=False),
              help='Directory to save preprocessed images.')
@click.option('--n-jobs', default=-1, type=int, help='Number of parallel jobs. Default is -1 (all CPUs).')
@click.option('--low-pass-sigma',
              default=0.03,
              type=float,
              help='Standard deviation for Gaussian low-pass filter in frequency domain applied to the phase correlation.')
@click.option('--mask-mode',
              type=click.Choice(MaskMode),
              default=MaskMode.AUTO_PER_IMAGE,
              help='Masking mode to use. "auto" determines the moon mask size automatically for each image. "max" '
                   'uses the maximum moon mask size across all images.')
@click.option('--mask-inner-radius',
              default=1.2,
              type=float,
              help='Inner radius of the annulus mask in multiples of the moon radius.')
@click.option('--mask-outer-radius',
              default=2.0,
              type=float,
              help='Outer radius of the annulus mask in multiples of the inner radius. Set to -1 to only mask the moon.')
@click.option('--save-preprocessed-post-alignment-images',
              is_flag=True,
              help='If set, saves preprocessed images after alignment. Useful for troubleshooting alignment issues.')
def align(reference_image: str,
          images_to_align: str,
          output_dir: str,
          n_jobs: int,
          low_pass_sigma: float,
          mask_mode: MaskMode,
          mask_inner_radius: float,
          mask_outer_radius: float,
          save_preprocessed_post_alignment_images: bool):
    """
    Align multiple eclipse images to reference image.
    """
    ref_image = preprocess_with_auto_mask(open_image(reference_image), mask_inner_radius, mask_outer_radius)

    click.echo(f"Processing {len(images_to_align)} images...")

    output_dir_abs = os.path.abspath(output_dir)
    click.echo(f'Writing aligned images to directory: {output_dir_abs}')

    # Ensure the output directory exists
    os.makedirs(output_dir_abs, exist_ok=True)

    if mask_mode == 'max':
        # TODO: Implement logic to determine the maximum moon mask size across all images
        pass

    # Process all images in parallel using joblib
    results = list(tqdm(total=len(images_to_align),
                        iterable=joblib.Parallel(n_jobs=n_jobs, prefer='threads', return_as='generator_unordered')(
                            joblib.delayed(_align_single_image)(ref_image, image_path, low_pass_sigma, output_dir_abs,
                                                                mask_inner_radius, mask_outer_radius,
                                                                save_preprocessed_post_alignment_images)
                            for image_path in images_to_align
                        )))

    results.sort(key=lambda x: x[0])  # Sort results by image path

    # Print results
    for image_path, scale, rotation_degrees, (translation_y, translation_x) in results:
        click.echo(f"Image: {image_path}")
        click.echo(f"  Scale: {scale:.4f}")
        click.echo(f"  Rotation: {rotation_degrees:.4f} degrees")
        click.echo(f"  Translation: {translation_x:.4f}, {translation_y:.4f}")


def _align_single_image(reference_image: np.ndarray,
                        image_path: str,
                        low_pass_sigma: float,
                        output_dir: str,
                        mask_inner_radius: float,
                        mask_outer_radius: float,
                        save_preprocessed_post_alignment_images: bool = False) -> \
        tuple[str, float, float, tuple[float, float]]:
    """
    Process a single image alignment operation.
    :param output_dir: Directory to save the output images
    :param reference_image: Preprocessed reference image data
    :param image_path: Path to the image file to align
    :param low_pass_sigma: Standard deviation for Gaussian low-pass filter in frequency domain applied to the phase correlation.
    :param mask_inner_radius: Inner radius of the annulus mask in multiples of the moon radius.
    :param mask_outer_radius: Outer radius of the annulus mask in multiples of the inner radius. Set to -1 to only mask the moon.
    :param save_preprocessed_post_alignment_images: If true, save preprocessed images after alignment.
    :return: Tuple of image path, scale, rotation in degrees, and translation vector (dy, dx)
    """
    raw_image = open_image(image_path)
    image_to_align = preprocess_with_auto_mask(raw_image, mask_inner_radius, mask_outer_radius)
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
    np.clip(aligned_image, 0.0, 1.0, out=aligned_image)  # Ensure values are in the range [0, 1]

    # Save the aligned image as a TIFF file
    orig_filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = os.path.join(output_dir, f"{orig_filename_without_ext}_aligned.tiff")
    save_tiff(aligned_image, output_filename)

    if save_preprocessed_post_alignment_images:
        # Normalize the image to have mean 0 and std 1, then shift to have mean 0.5, so it is easier to view in an external program
        preproc_norm = np.clip((image_to_align - image_to_align.mean()) / image_to_align.std() + 0.5,
                               0.0,
                               1.0,
                               dtype=np.float32)
        aligned_preproc = cv2.warpAffine(preproc_norm,
                                         transform_matrix,
                                         (preproc_norm.shape[1], preproc_norm.shape[0]),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
        preproc_aligned_filename = os.path.join(output_dir, f"{orig_filename_without_ext}_aligned_preproc.tiff")
        save_tiff(aligned_preproc, preproc_aligned_filename)

    return image_path, float(scale), float(rotation_degrees), (float(translation_y), float(translation_x))
