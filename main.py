import sys

import matplotlib.pyplot as plt
import numpy as np
import rawpy
import scipy.ndimage

import preprocessing.filtering
import preprocessing.masking
import utils.circlefinder
from utils.circlefinder import DetectedCircle


def open_raw_image(path: str) -> np.ndarray:
    with rawpy.imread(path) as raw:
        return np.float32(
            raw.postprocess(
                output_bps=16,
                user_flip=0,
                gamma=(1.0, 1.0),
                user_wb=[1.0, 1.0, 1.0, 1.0],
                output_color=rawpy.ColorSpace.raw,
                no_auto_bright=True)) / 65535.0


def main(args):
    raw_image = open_raw_image(args[1])
    image = np.mean(raw_image, axis=2)

    moon = utils.circlefinder.find_circle(image, min_radius=400, max_radius=700)
    if moon is None:
        print('Could not find moon in image')
        return
    else:
        print(f'Found moon at {moon.center} with radius {moon.radius}')

    saturated_radius = estimate_saturated_radius(moon, raw_image)
    moon_mask_radius = 1.2 * (saturated_radius if saturated_radius else moon.radius)
    image_for_alignment = preprocess(image, moon.center, moon_mask_radius)

    plt.imshow(image_for_alignment, cmap='gray')
    plt.title('Image ready for phase-correlation registration')
    plt.show()


def estimate_saturated_radius(moon_params: DetectedCircle, raw_image: np.ndarray) -> float | None:
    saturated_pixels = scipy.ndimage.median_filter(np.max(raw_image, axis=2), size=3) > 0.999
    saturated_radius = None
    if np.any(saturated_pixels):
        distances = np.linalg.norm(np.argwhere(saturated_pixels) - moon_params.center, axis=1)
        saturated_radius = np.max(distances)
        print(f'Radius of saturated area {saturated_radius}')
    else:
        print('No saturated pixels found')
    return saturated_radius


def preprocess(image: np.ndarray, moon_center: tuple, moon_mask_radius: float) -> np.ndarray:
    window_mask = preprocessing.masking.hann_window_mask(image.shape)
    moon_mask = preprocessing.masking.circle_mask(image, moon_center, moon_mask_radius)
    mask = window_mask * moon_mask
    filtered_image = image - preprocessing.filtering.rotational_blur(image, max_angle=2, center=moon_center)
    image_for_alignment = mask * filtered_image
    return image_for_alignment


if __name__ == '__main__':
    main(sys.argv)
