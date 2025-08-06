import os

import click
import cv2
import joblib
import numpy as np
from tqdm import tqdm

from eclipsetools.alignment import find_transform
from eclipsetools.common.circle_finder import find_circle, DetectedCircle
from eclipsetools.common.image_reader import open_image
from eclipsetools.common.image_writer import save_tiff
from eclipsetools.preprocessing.masking import MaskMode, find_max_mask_inner_radius
from eclipsetools.preprocessing.workflows import (
    preprocess_with_auto_mask,
    preprocess_with_fixed_mask,
)


@click.group()
def align():
    """
    Commands for aligning eclipse images.
    """
    pass


@align.command("corona")
@click.argument("reference_image", type=click.Path(exists=True))
@click.argument("images_to_align", nargs=-1, required=True)
@click.option(
    "--output-dir",
    default="output",
    type=click.Path(exists=False),
    help="Directory to save preprocessed images.",
)
@click.option(
    "--n-jobs",
    default=-1,
    type=int,
    help="Number of parallel jobs. Default is -1 (all CPUs).",
)
@click.option(
    "--low-pass-sigma",
    default=0.03,
    type=float,
    help="Standard deviation for Gaussian low-pass filter in frequency domain applied to the phase correlation.",
)
@click.option(
    "--mask-mode",
    type=click.Choice(MaskMode),
    default=MaskMode.AUTO_PER_IMAGE,
    help='Masking mode to use. "auto" determines the moon mask size automatically for each image. "max" '
    "uses the maximum moon mask size across all images.",
)
@click.option(
    "--mask-inner-radius",
    default=1.2,
    type=float,
    help="Inner radius of the annulus mask in multiples of the moon radius.",
)
@click.option(
    "--mask-outer-radius",
    default=2.0,
    type=float,
    help="Outer radius of the annulus mask in multiples of the inner radius. Set to -1 to only mask the moon.",
)
@click.option(
    "--save-preprocessed-post-alignment-images",
    is_flag=True,
    help="If set, saves preprocessed images after alignment. Useful for troubleshooting alignment issues.",
)
@click.option(
    "--crop",
    default=0,
    type=int,
    help="Crop this many pixels from each side during preprocessing. This helps dealing with artifacts "
    "caused by stacking several images and using the stack as a reference image.",
)
@click.option(
    "--moon-min-radius",
    default=200,
    type=int,
    help="Minimum moon radius in pixels for moon detection.",
)
@click.option(
    "--moon-max-radius",
    default=2000,
    type=int,
    help="Maximum moon radius in pixels for moon detection.",
)
def align_by_corona(
    reference_image: str,
    images_to_align: list[str],
    output_dir: str,
    n_jobs: int,
    low_pass_sigma: float,
    mask_mode: MaskMode,
    mask_inner_radius: float,
    mask_outer_radius: float,
    moon_min_radius: int,
    moon_max_radius: int,
    save_preprocessed_post_alignment_images: bool,
    crop: int,
):
    """
    Align images based on coronal details.
    """

    click.echo(f"Processing {len(images_to_align)} images...")

    output_dir_abs = os.path.abspath(output_dir)
    click.echo(f"Writing aligned images to directory: {output_dir_abs}")

    # Ensure the output directory exists
    os.makedirs(output_dir_abs, exist_ok=True)

    max_mask_inner_radius_px = None
    if mask_mode == "max":
        images_for_masking = [reference_image] + list(images_to_align)
        max_mask_inner_radius_px = find_max_mask_inner_radius(
            images_for_masking,
            mask_inner_radius,
            n_jobs,
            moon_min_radius,
            moon_max_radius,
            True,
        )
        ref_image = preprocess_with_fixed_mask(
            open_image(reference_image),
            max_mask_inner_radius_px,
            mask_outer_radius,
            crop,
            moon_min_radius,
            moon_max_radius,
        )
    else:
        ref_image = preprocess_with_auto_mask(
            open_image(reference_image),
            mask_inner_radius,
            mask_outer_radius,
            crop,
            moon_min_radius,
            moon_max_radius,
        )

    # Process all images in parallel using joblib
    results = list(
        tqdm(
            total=len(images_to_align),
            desc="Aligning images",
            unit="img",
            iterable=joblib.Parallel(
                n_jobs=n_jobs, prefer="threads", return_as="generator_unordered"
            )(
                joblib.delayed(_align_single_image_by_corona)(
                    ref_image,
                    image_path,
                    low_pass_sigma,
                    output_dir_abs,
                    mask_inner_radius,
                    mask_outer_radius,
                    max_mask_inner_radius_px,
                    save_preprocessed_post_alignment_images,
                    crop,
                    moon_min_radius,
                    moon_max_radius,
                )
                for image_path in images_to_align
            ),
        )
    )

    results.sort(key=lambda x: x[0])  # Sort results by image path

    # Print results
    for image_path, scale, rotation_degrees, (translation_y, translation_x) in results:
        click.echo(f"Image: {image_path}")
        click.echo(f"  Scale: {scale:.4f}")
        click.echo(f"  Rotation: {rotation_degrees:.4f} degrees")
        click.echo(f"  Translation: {translation_x:.4f}, {translation_y:.4f}")


def _align_single_image_by_corona(
    reference_image: np.ndarray,
    image_path: str,
    low_pass_sigma: float,
    output_dir: str,
    mask_inner_radius: float,
    mask_outer_radius: float,
    mask_inner_radius_px: float | None,
    save_preprocessed_post_alignment_images: bool,
    crop: int,
    moon_min_radius: int,
    moon_max_radius: int,
) -> tuple[str, float, float, tuple[float, float]]:
    """
    Process a single image alignment operation.
    :param output_dir: Directory to save the output images
    :param reference_image: Preprocessed reference image data
    :param image_path: Path to the image file to align
    :param low_pass_sigma: Standard deviation for Gaussian low-pass filter in frequency domain applied to the phase correlation.
    :param mask_inner_radius: Inner radius of the annulus mask in multiples of the moon radius.
    :param mask_outer_radius: Outer radius of the annulus mask in multiples of the inner radius. Set to -1 to only mask the moon.
    :param mask_inner_radius_px: Inner radius of the annulus mask in pixels. If None, use the auto mask.
    :param save_preprocessed_post_alignment_images: If true, save preprocessed images after alignment.
    :param crop: Number of pixels to crop from each side during preprocessing.
    :param moon_min_radius: Minimum moon radius in pixels for moon detection.
    :param moon_max_radius: Maximum moon radius in pixels for moon detection.
    :return: Tuple of image path, scale, rotation in degrees, and translation vector (dy, dx)
    """
    rgb_image = open_image(image_path)
    if mask_inner_radius_px is None:
        image_to_align = preprocess_with_auto_mask(
            rgb_image,
            mask_inner_radius,
            mask_outer_radius,
            crop,
            moon_min_radius,
            moon_max_radius,
        )
    else:
        image_to_align = preprocess_with_fixed_mask(
            rgb_image,
            mask_inner_radius_px,
            mask_outer_radius,
            crop,
            moon_min_radius,
            moon_max_radius,
        )
    # TODO: Parametrize scale difference allowance
    scale, rotation_degrees, (translation_y, translation_x) = find_transform(
        reference_image, image_to_align, low_pass_sigma, allow_scale=False
    )

    transform_matrix = _get_transform_matrix(
        1.0 / scale,
        -rotation_degrees,
        (rgb_image.shape[0] / 2, rgb_image.shape[1] / 2),
        (-translation_y, -translation_x),
    )
    aligned_image = cv2.warpAffine(
        rgb_image,
        transform_matrix,
        (rgb_image.shape[1], rgb_image.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    np.clip(
        aligned_image, 0.0, 1.0, out=aligned_image
    )  # Ensure values are in the range [0, 1]

    # Save the aligned image as a TIFF file
    orig_filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = os.path.join(
        output_dir, f"{orig_filename_without_ext}_aligned.tiff"
    )
    save_tiff(aligned_image, output_filename)

    if save_preprocessed_post_alignment_images:
        # Normalize the image to have mean 0 and std 1, then shift to have mean 0.5, so it is easier to view in an external program
        preproc_norm = np.clip(
            (image_to_align - image_to_align.mean()) / image_to_align.std() + 0.5,
            0.0,
            1.0,
            dtype=np.float32,
        )
        aligned_preproc = cv2.warpAffine(
            preproc_norm,
            transform_matrix,
            (preproc_norm.shape[1], preproc_norm.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        preproc_aligned_filename = os.path.join(
            output_dir, f"{orig_filename_without_ext}_aligned_preproc.tiff"
        )
        save_tiff(aligned_preproc, preproc_aligned_filename)

    return (
        image_path,
        float(scale),
        float(rotation_degrees),
        (float(translation_y), float(translation_x)),
    )


def _get_transform_matrix(
    scale: float,
    rotation_degrees: float,
    rotation_center: tuple[float, float],
    translation: tuple[float, float],
) -> np.ndarray:
    """
    Get the transformation matrix for the given scale, rotation, and translation.
    The transformation first scales and rotates the image around the specified center, and then translates it.
    :param scale: Scale factor
    :param rotation_degrees: Rotation in degrees
    :param rotation_center: Center of rotation (cy, cx)
    :param translation: Translation vector (ty, tx)
    :return: 2x3 transformation matrix
    """
    rotation_scale_matrix = np.vstack(
        [
            cv2.getRotationMatrix2D(
                rotation_center,
                rotation_degrees,
                scale,
            ),
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    translation_matrix = np.array(
        [[1, 0, translation[1]], [0, 1, translation[0]], [0, 0, 1]], dtype=np.float32
    )

    transform_matrix = (rotation_scale_matrix @ translation_matrix)[:2, :]
    return transform_matrix


@align.command("moon")
@click.argument("reference_image", type=click.Path(exists=True))
@click.argument("images_to_align", nargs=-1, required=True)
@click.option(
    "--moon-min-radius",
    default=200,
    type=int,
    help="Minimum moon radius in pixels for moon detection.",
)
@click.option(
    "--moon-max-radius",
    default=2000,
    type=int,
    help="Maximum moon radius in pixels for moon detection.",
)
@click.option(
    "--output-dir",
    default="output",
    type=click.Path(exists=False),
    help="Directory to save preprocessed images.",
)
@click.option(
    "--n-jobs",
    default=-1,
    type=int,
    help="Number of parallel jobs. Default is -1 (all CPUs).",
)
def align_by_moon(
    reference_image: str,
    images_to_align: list[str],
    moon_min_radius: int,
    moon_max_radius: int,
    output_dir: str,
    n_jobs: int,
):
    """
    Align images based on the moon's position.
    """
    click.echo(f"Processing {len(images_to_align)} images...")

    output_dir_abs = os.path.abspath(output_dir)
    click.echo(f"Writing aligned images to directory: {output_dir_abs}")

    # Ensure the output directory exists
    os.makedirs(output_dir_abs, exist_ok=True)

    ref_moon_params = find_circle(
        open_image(reference_image)[:, :, 1], moon_min_radius, moon_max_radius
    )

    _ = list(
        tqdm(
            desc="Aligning images",
            unit="img",
            total=len(images_to_align),
            iterable=joblib.Parallel(
                prefer="threads", return_as="generator_unordered", n_jobs=n_jobs
            )(
                joblib.delayed(_align_single_image_by_moon)(
                    image_path,
                    ref_moon_params,
                    output_dir_abs,
                    moon_min_radius,
                    moon_max_radius,
                )
                for image_path in images_to_align
            ),
        )
    )


def _align_single_image_by_moon(
    image_path: str,
    ref_moon_params: DetectedCircle,
    output_dir: str,
    moon_min_radius: int,
    moon_max_radius: int,
):

    image = open_image(image_path)
    moon_params = find_circle(image[:, :, 1], moon_min_radius, moon_max_radius)

    if moon_params is None:
        click.echo(f"Moon not found in {image_path}. Skipping alignment.")
        return

    translation_x = ref_moon_params.center[1] - moon_params.center[1]
    translation_y = ref_moon_params.center[0] - moon_params.center[0]

    transform_matrix = _get_transform_matrix(
        1.0,
        0.0,
        (0, 0),
        (translation_y, translation_x),
    )
    aligned_image = cv2.warpAffine(
        image,
        transform_matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    np.clip(aligned_image, 0.0, 1.0, out=aligned_image)

    # Save the aligned image as a TIFF file
    orig_filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = os.path.join(
        output_dir, f"{orig_filename_without_ext}_aligned.tiff"
    )
    save_tiff(aligned_image, output_filename)


@align.command()
@click.argument("images_to_preprocess", nargs=-1, required=True)
@click.option(
    "--n-jobs",
    default=-1,
    type=int,
    help="Number of parallel jobs. Default is -1 (all CPUs).",
)
@click.option(
    "--output-dir",
    default="output",
    type=click.Path(exists=False),
    help="Directory to save preprocessed images.",
)
@click.option(
    "--moon-min-radius",
    default=200,
    type=int,
    help="Minimum moon radius in pixels for moon detection.",
)
@click.option(
    "--moon-max-radius",
    default=2000,
    type=int,
    help="Maximum moon radius in pixels for moon detection.",
)
@click.option(
    "--mask-mode",
    type=click.Choice(MaskMode),
    default=MaskMode.AUTO_PER_IMAGE,
    help='Masking mode to use. "auto" determines the moon mask size automatically for each image. "max" '
    "uses the maximum moon mask size across all images.",
)
@click.option(
    "--mask-inner-radius",
    default=1.2,
    type=float,
    help="Inner radius of the annulus mask in multiples of " "the moon radius.",
)
@click.option(
    "--mask-outer-radius",
    default=2.0,
    type=float,
    help="Outer radius of the annulus mask in multiples of "
    "the inner radius. Set to -1 to only mask the moon.",
)
def preprocess_only(
    images_to_preprocess: list[str],
    n_jobs: int,
    output_dir: str,
    moon_min_radius: int,
    moon_max_radius: int,
    mask_mode: MaskMode,
    mask_inner_radius: float,
    mask_outer_radius: float,
):
    """
    Preprocess images for alignment. This is useful for testing and troubleshooting preprocessing settings without
    doing actual alignment. The output will be 32-bit grayscale TIFF images, with a mean of 0.5 and a standard deviation of 1.0,
    which can be viewed in an external program like Adobe Photoshop.
    """

    click.echo(f"Preprocessing {len(images_to_preprocess)} images...")

    output_dir_abs = os.path.abspath(output_dir)
    click.echo("Writing preprocessed images to directory: " + output_dir_abs)

    # Ensure the output directory exists
    os.makedirs(output_dir_abs, exist_ok=True)

    max_mask_inner_radius_px = None
    if mask_mode == "max":
        max_mask_inner_radius_px = find_max_mask_inner_radius(
            images_to_preprocess,
            mask_inner_radius,
            n_jobs,
            moon_min_radius,
            moon_max_radius,
            True,
        )

    list(
        tqdm(
            total=len(images_to_preprocess),
            desc="Preprocessing images",
            unit="img",
            iterable=Parallel(
                n_jobs=n_jobs, prefer="threads", return_as="generator_unordered"
            )(
                joblib.delayed(_open_and_preprocess)(
                    path,
                    output_dir_abs,
                    mask_inner_radius,
                    mask_outer_radius,
                    max_mask_inner_radius_px,
                    moon_min_radius,
                    moon_max_radius,
                )
                for path in images_to_preprocess
            ),
        )
    )


def _open_and_preprocess(
    image_path: str,
    output_dir: str,
    mask_inner_radius_multiplier: float,
    mask_outer_radius_multiplier: float,
    mask_inner_radius_px: float | None,
    moon_min_radius: int,
    moon_max_radius: int,
):
    rgb_image = open_image(image_path)
    if mask_inner_radius_px is None:
        image_preproc = preprocess_with_auto_mask(
            rgb_image,
            mask_inner_radius_multiplier,
            mask_outer_radius_multiplier,
            0,
            moon_min_radius,
            moon_max_radius,
        )
    else:
        image_preproc = preprocess_with_fixed_mask(
            rgb_image,
            mask_inner_radius_px,
            mask_outer_radius_multiplier,
            0,
            moon_min_radius,
            moon_max_radius,
        )

    # Normalize the image to have mean 0 and std 1, then shift to have mean 0.5, so it is easier to view in an external program
    image_preproc = np.clip(
        (image_preproc - image_preproc.mean()) / image_preproc.std() + 0.5, 0.0, 1.0
    )

    orig_filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = os.path.join(
        output_dir, f"{orig_filename_without_ext}_preproc.tiff"
    )
    save_tiff(image_preproc, output_filename)
