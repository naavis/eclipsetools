import os

import click
import joblib
import numpy as np
from joblib import Parallel
from tqdm import tqdm

from eclipsetools.preprocessing.workflows import preprocess_with_auto_mask, preprocess_with_fixed_mask
from eclipsetools.preprocessing.masking import MaskMode, find_mask_inner_radius_px
from eclipsetools.utils.image_reader import open_image
from eclipsetools.utils.image_writer import save_tiff


@click.command()
@click.argument('images_to_preprocess', nargs=-1, required=True)
@click.option('--n-jobs', default=-1, type=int, help='Number of parallel jobs. Default is -1 (all CPUs).')
@click.option('--output-dir', default='output', type=click.Path(exists=False),
              help='Directory to save preprocessed images.')
@click.option('--mask-mode',
              type=click.Choice(MaskMode),
              default=MaskMode.AUTO_PER_IMAGE,
              help='Masking mode to use. "auto" determines the moon mask size automatically for each image. "max" '
                   'uses the maximum moon mask size across all images.')
@click.option('--mask-inner-radius', default=1.2, type=float, help='Inner radius of the annulus mask in multiples of '
                                                                   'the moon radius.')
@click.option('--mask-outer-radius', default=2.0, type=float, help='Outer radius of the annulus mask in multiples of '
                                                                   'the inner radius. Set to -1 to only mask the moon.')
def preprocess_only(images_to_preprocess: list[str],
                    n_jobs: int,
                    output_dir: str,
                    mask_mode: MaskMode,
                    mask_inner_radius: float,
                    mask_outer_radius: float):
    """
    Preprocess images for alignment. The output will be 32-bit grayscale TIFF images, with both negative and positive values.
    """

    click.echo(f"Preprocessing {len(images_to_preprocess)} images...")

    output_dir_abs = os.path.abspath(output_dir)
    click.echo('Writing preprocessed images to directory: ' + output_dir_abs)

    # Ensure the output directory exists
    os.makedirs(output_dir_abs, exist_ok=True)

    max_mask_inner_radius_px = None
    if mask_mode == 'max':
        max_mask_inner_radius_px = _find_max_mask_inner_radius(images_to_preprocess, mask_inner_radius, n_jobs)

    list(
        tqdm(total=len(images_to_preprocess),
             desc='Preprocessing images',
             unit='img',
             iterable=Parallel(n_jobs=n_jobs, prefer='threads', return_as='generator_unordered')(
                 joblib.delayed(_open_and_preprocess)(path,
                                                      output_dir_abs,
                                                      mask_inner_radius,
                                                      mask_outer_radius,
                                                      max_mask_inner_radius_px) for
                 path in
                 images_to_preprocess)))


def _find_max_mask_inner_radius(images: list[str], inner_multiplier: float, n_jobs: int) -> float:
    """
    Find the maximum inner radius in pixels for the moon mask across all images.
    :param images: Paths to images
    :param inner_multiplier: Number to multiply the found moon radius by to get the mask radius in pixels.
    :param n_jobs: Number of parallel jobs to use for processing.
    :return: Biggest mask radius that covers the moon and saturated areas in all images.
    """
    jobs = [joblib.delayed(find_mask_inner_radius_px)(image_path, inner_multiplier) for image_path in
            images]
    parallel = Parallel(n_jobs=n_jobs, prefer='threads', return_as='generator_unordered')
    inner_radii = list(tqdm(total=len(images), iterable=parallel(jobs), desc='Finding suitable moon mask size',
                            unit='img'))
    return np.max(inner_radii)


def _open_and_preprocess(image_path: str,
                         output_dir: str,
                         mask_inner_radius_multiplier: float,
                         mask_outer_radius_multiplier: float,
                         mask_inner_radius_px: float | None = None):
    rgb_image = open_image(image_path)
    if mask_inner_radius_px is None:
        image_preproc = preprocess_with_auto_mask(rgb_image, mask_inner_radius_multiplier, mask_outer_radius_multiplier)
    else:
        image_preproc = preprocess_with_fixed_mask(rgb_image, mask_inner_radius_px, mask_outer_radius_multiplier)

    # Normalize the image to have mean 0 and std 1, then shift to have mean 0.5, so it is easier to view in an external program
    image_preproc = np.clip((image_preproc - image_preproc.mean()) / image_preproc.std() + 0.5, 0.0, 1.0)

    orig_filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = os.path.join(output_dir, f"{orig_filename_without_ext}_preproc.tiff")
    save_tiff(image_preproc, output_filename)
