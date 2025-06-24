import cv2
import joblib
import numpy as np

from eclipsetools import preprocessing
from eclipsetools.alignment import find_translation, find_transform
from eclipsetools.utils.raw_reader import open_raw_image


def test_translate():
    ref_image = open_raw_image(r'tests\images\eclipse_5ms.CR3')
    ref_image_preproc = preprocessing.preprocess_for_alignment(ref_image)

    num_tests = 10
    rng = np.random.default_rng(122807528840384100672342137672332424406)
    offsets = rng.uniform(-40.0, 40.0, (num_tests, 2))
    # We can reuse the same noise in each test image to avoid passing the RNG around to several threads
    noise_image = rng.normal(0.0, 0.001, ref_image.shape)
    errors = joblib.Parallel(n_jobs=-1, prefer='threads')(
        joblib.delayed(_find_test_image_translation)(ref_image, ref_image_preproc, offset, noise_image) for offset in
        offsets)

    for error in errors:
        assert error is not None and error < 0.2


def _find_test_image_translation(ref_image: np.ndarray,
                                 ref_image_preproc: np.ndarray,
                                 offset: np.ndarray,
                                 noise_image: np.ndarray) -> float:
    offset_y, offset_x = offset
    translation_matrix = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y]
    ], dtype=np.float32)
    test_image = cv2.warpAffine(
        ref_image,
        translation_matrix,
        dsize=(ref_image.shape[1], ref_image.shape[0]))
    # We add some Gaussian noise to the translated image to simulate varying noise in real images
    noisy_test_image = np.clip(test_image + noise_image, 0.0, 1.0)
    translated_test_image = preprocessing.preprocess_for_alignment(noisy_test_image)
    found_translation = find_translation(ref_image_preproc, translated_test_image, low_pass_sigma=0.2)
    error = np.sqrt(np.sum(np.square(found_translation - offset)))
    return error


def test_align():
    print('')
    ref_image = open_raw_image(r'tests\images\eclipse_5ms.CR3')
    num_tests = 10
    rng = np.random.default_rng(122807528840384100672342137672332424406)
    offsets = rng.uniform(-20.0, 20.0, (num_tests, 2))
    rotations = rng.uniform(-10.0, 10.0, num_tests)
    scales = rng.uniform(0.8, 1.2, num_tests)
    # We can reuse the same noise in each test image to avoid passing the RNG around to several threads
    noise_image = rng.normal(0.0, 0.001, ref_image.shape)
    # noise_image = np.zeros_like(ref_image, dtype=np.float32)  # No noise for now
    errors = joblib.Parallel(n_jobs=1, prefer='threads')(
        joblib.delayed(_find_transform_error)(
            ref_image,
            offset,
            rotation,
            scale,
            noise_image) for (offset, rotation, scale) in zip(offsets, rotations, scales))

    for error in errors:
        (scale_error, rotation_error, translation_error) = error
        print(
            f'Scale error: {scale_error:.2f}, rotation error: {rotation_error:.2f}, translation error: {translation_error:.2f}')
        # assert scale_error < 0.05, f'Scale error too high: {scale_error}'
        # assert rotation_error < 0.3, f'Rotation error too high: {rotation_error}'
        # assert translation_error < 0.2, f'Translation error too high: {translation_error}'


def _find_transform_error(ref_image: np.ndarray,
                          offset: np.ndarray,
                          rotation: float,
                          scale: float,
                          noise_image: np.ndarray) -> tuple[float, float, float]:
    # We crop images to avoid artifacts at the edges that can occur due to the affine transformation
    crop_margin = 500
    # Cropping has to be done before preprocessing to avoid artifacts at the edges
    ref_image_preproc = preprocessing.preprocess_for_alignment(
        ref_image[crop_margin:-crop_margin, crop_margin:-crop_margin, :])

    test_image = transform_image(ref_image, offset, rotation, scale)
    noisy_test_image = np.clip(test_image + noise_image, 0.0, 1.0, dtype=np.float32)

    preproc_image = preprocessing.preprocess_for_alignment(
        noisy_test_image[crop_margin:-crop_margin, crop_margin:-crop_margin, :])

    recovered_scale, recovered_rotation, recovered_translation = find_transform(ref_image_preproc, preproc_image, 0.2)
    print()
    print(f'Expected scale {scale:.2f}, rotation {rotation:.2f}, offset {offset}')
    print(
        f'Recovered scale {recovered_scale:.2f}, rotation {recovered_rotation:.2f}, translation {recovered_translation}')
    return (abs(1.0 - recovered_scale / scale),
            abs(rotation - recovered_rotation),
            np.sqrt(np.sum(np.square(recovered_translation - offset))))


def transform_image(image, translation, rotation, scale):
    translate_y, translate_x = translation

    translation_matrix = np.array([[1, 0, translate_x],
                                   [0, 1, translate_y],
                                   [0, 0, 1]], dtype=np.float32)

    # Get image center
    center = (image.shape[1] // 2, image.shape[0] // 2)

    # Create rotation matrix around center with scaling
    transform_matrix = np.vstack((cv2.getRotationMatrix2D(center=center, angle=rotation, scale=scale), [0, 0, 1]),
                                 dtype=np.float32)

    # Create affine matrix that translates first, then rotates and scales
    matrix = (transform_matrix @ translation_matrix)[:2, :]

    transformed_image = cv2.warpAffine(
        image,
        matrix,
        dsize=(image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=[0, 0, 0]).astype(np.float32)

    return transformed_image
