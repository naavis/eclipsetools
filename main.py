import sys

import preprocessing.filtering
import preprocessing.masking
import utils.circlefinder
import rawpy
import numpy as np

import matplotlib.pyplot as plt


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
    image = np.mean(open_raw_image(args[1]), axis=2)

    moon = utils.circlefinder.find_circle(image, min_radius=300, max_radius=700)

    window_mask = preprocessing.masking.hann_window_mask(image.shape)
    # TODO: The moon mask radius should be parametrized, so it is larger for more over-exposed images
    moon_mask = preprocessing.masking.circle_mask(image, moon.center, 1.4 * moon.radius)
    mask = window_mask * moon_mask

    filtered_image = image - preprocessing.filtering.rotational_blur(image, max_angle=2, center=moon.center)

    image_for_alignment = mask * filtered_image
    plt.imshow(image_for_alignment, cmap='gray')
    plt.title('Image ready for phase-correlation registration')
    plt.show()



if __name__ == '__main__':
    main(sys.argv)
