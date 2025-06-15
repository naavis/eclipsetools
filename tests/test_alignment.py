import cv2
import numpy as np

import preprocessing
import utils.raw_reader
from alignment import find_translation


def test_translate():
    ref_image = utils.raw_reader.open_raw_image(r'images\eclipse_5ms.CR3')
    ref_image_preproc = preprocessing.preprocess_for_alignment(ref_image[50:-50, 50:-50, :])

    num_tests = 10
    rng = np.random.default_rng(122807528840384100672342137672332424406)
    offsets = rng.uniform(-40.0, 40.0, (num_tests, 2))
    for offset in offsets:
        offset_y, offset_x = offset

        translation_matrix = np.array([
            [1, 0, offset_x],
            [0, 1, offset_y]
        ], dtype=np.float32)
        translated_image = cv2.warpAffine(
            ref_image,
            translation_matrix,
            dsize=(ref_image.shape[1], ref_image.shape[0]))

        # We add some Gaussian noise to the translated image to simulate varying noise in real images
        translated_image = np.clip(translated_image + rng.normal(0.0, 0.01, translated_image.shape), 0.0, 1.0)

        translated_image_preproc = preprocessing.preprocess_for_alignment(translated_image[50:-50, 50:-50, :])
        found_translation = find_translation(ref_image_preproc, translated_image_preproc)

        error = np.sqrt(np.sum(np.square(found_translation - offset)))
        print(f"Total error for offset x: {offset_x}, y: {offset_y} is {error}")
        assert error < 0.2
