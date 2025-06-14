import cv2
import numpy as np

import preprocessing
import utils.raw_reader
from alignment import find_translation


def test_single_translate():
    ref_image = utils.raw_reader.open_raw_image(r'images\eclipse_5ms.CR3')
    offset_y = 42.1
    offset_x = -12.5

    translation_matrix = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y]
    ], dtype=np.float32)
    translated_image = cv2.warpAffine(ref_image, translation_matrix, dsize=(ref_image.shape[1], ref_image.shape[0]))

    ref_image_preproc = preprocessing.preprocess_for_alignment(ref_image[50:-50, 50:-50, :])
    translated_image_preproc = preprocessing.preprocess_for_alignment(translated_image[50:-50, 50:-50, :])

    found_translation = find_translation(ref_image_preproc, translated_image_preproc)

    total_error = np.sqrt(np.sum(np.square(found_translation - np.array([offset_y, offset_x]))))
    assert total_error < 0.5
