import cv2
import joblib
import numpy as np

from eclipsetools import preprocessing
from eclipsetools.alignment import find_translation
from eclipsetools.utils.raw_reader import open_raw_image


def test_translate():
    ref_image = open_raw_image(r'tests\images\eclipse_5ms.CR3')
    ref_image_preproc = preprocessing.preprocess_for_alignment(ref_image)

    num_tests = 10
    rng = np.random.default_rng(122807528840384100672342137672332424406)
    offsets = rng.uniform(-40.0, 40.0, (num_tests, 2))
    # We can reuse the same noise in each test image to avoid passing the RNG around to several threads
    noise_image = rng.normal(0.0, 0.01, ref_image.shape)
    errors = joblib.Parallel(n_jobs=-1, prefer='threads')(
        joblib.delayed(align_test_image)(ref_image, ref_image_preproc, offset, noise_image) for offset in offsets)

    for error in errors:
        assert error < 0.2


def align_test_image(ref_image: np.ndarray,
                     ref_image_preproc: np.ndarray,
                     offset: np.ndarray,
                     noise_image: np.ndarray) -> float:
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
    translated_image = np.clip(translated_image + noise_image, 0.0, 1.0)
    translated_image_preproc = preprocessing.preprocess_for_alignment(translated_image)
    found_translation = find_translation(ref_image_preproc, translated_image_preproc, low_pass_sigma=0.2)
    error = np.sqrt(np.sum(np.square(found_translation - offset)))
    return error
