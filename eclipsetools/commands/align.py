import os

import click
import cv2
import joblib
import numpy as np
from tqdm import tqdm

from eclipsetools.alignment import find_transform
from eclipsetools.preprocessing.masking import MaskMode, find_mask_inner_radius_px
from eclipsetools.preprocessing.workflows import (
    preprocess_with_auto_mask,
    preprocess_with_fixed_mask,
)
from eclipsetools.utils.image_reader import open_image
from eclipsetools.utils.image_writer import save_tiff


@click.command()
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
def align(
    reference_image: str,
    images_to_align: list[str],
    output_dir: str,
    n_jobs: int,
    low_pass_sigma: float,
    mask_mode: MaskMode,
    mask_inner_radius: float,
    mask_outer_radius: float,
    save_preprocessed_post_alignment_images: bool,
    crop: int,
):
    """
    Align multiple eclipse images to reference image.
    """

    max_mask_inner_radius_px = None
    if mask_mode == "max":
        images_for_masking = [reference_image] + list(images_to_align)
        max_mask_inner_radius_px = _find_max_mask_inner_radius(
            images_for_masking, mask_inner_radius, n_jobs
        )
        ref_image = preprocess_with_fixed_mask(
            open_image(reference_image),
            max_mask_inner_radius_px,
            mask_outer_radius,
            crop,
        )
    else:
        ref_image = preprocess_with_auto_mask(
            open_image(reference_image), mask_inner_radius, mask_outer_radius, crop
        )

    click.echo(f"Processing {len(images_to_align)} images...")

    output_dir_abs = os.path.abspath(output_dir)
    click.echo(f"Writing aligned images to directory: {output_dir_abs}")

    # Ensure the output directory exists
    os.makedirs(output_dir_abs, exist_ok=True)

    # Process all images in parallel using joblib
    results = list(
        tqdm(
            total=len(images_to_align),
            desc="Aligning images",
            unit="img",
            iterable=joblib.Parallel(
                n_jobs=n_jobs, prefer="threads", return_as="generator_unordered"
            )(
                joblib.delayed(_align_single_image)(
                    ref_image,
                    image_path,
                    low_pass_sigma,
                    output_dir_abs,
                    mask_inner_radius,
                    mask_outer_radius,
                    max_mask_inner_radius_px,
                    save_preprocessed_post_alignment_images,
                    crop,
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


def _align_single_image(
    reference_image: np.ndarray,
    image_path: str,
    low_pass_sigma: float,
    output_dir: str,
    mask_inner_radius: float,
    mask_outer_radius: float,
    mask_inner_radius_px: float | None = None,
    save_preprocessed_post_alignment_images: bool = False,
    crop=0,
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
    :return: Tuple of image path, scale, rotation in degrees, and translation vector (dy, dx)
    """
    raw_image = open_image(image_path)
    if mask_inner_radius_px is None:
        image_to_align = preprocess_with_auto_mask(
            raw_image, mask_inner_radius, mask_outer_radius, crop
        )
    else:
        image_to_align = preprocess_with_fixed_mask(
            raw_image, mask_inner_radius_px, mask_outer_radius, crop
        )
    scale, rotation_degrees, (translation_y, translation_x) = find_transform(
        reference_image, image_to_align, low_pass_sigma, allow_scale=False
    )

    rotation_scale_matrix = np.vstack(
        [
            cv2.getRotationMatrix2D(
                (raw_image.shape[1] / 2, raw_image.shape[0] / 2),
                -rotation_degrees,
                1.0 / scale,
            ),
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    translation_matrix = np.array(
        [[1, 0, -translation_x], [0, 1, -translation_y], [0, 0, 1]], dtype=np.float32
    )

    transform_matrix = (rotation_scale_matrix @ translation_matrix)[:2, :]
    aligned_image = cv2.warpAffine(
        raw_image,
        transform_matrix,
        (raw_image.shape[1], raw_image.shape[0]),
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


def _find_max_mask_inner_radius(
    images: list[str], inner_multiplier: float, n_jobs: int
) -> float:
    """
    Find the maximum inner radius in pixels for the moon mask across all images.
    :param images: Paths to images
    :param inner_multiplier: Number to multiply the found moon radius by to get the mask radius in pixels.
    :param n_jobs: Number of parallel jobs to use for processing.
    :return: Biggest mask radius that covers the moon and saturated areas in all images.
    """
    jobs = [
        joblib.delayed(find_mask_inner_radius_px)(image_path, inner_multiplier)
        for image_path in images
    ]
    parallel = joblib.Parallel(
        n_jobs=n_jobs, prefer="threads", return_as="generator_unordered"
    )
    inner_radii = tqdm(
        total=len(images),
        iterable=parallel(jobs),
        desc="Finding suitable moon mask size",
        unit="img",
    )
    return max(inner_radii)
