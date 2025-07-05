import cv2
import numpy as np

from eclipsetools import preprocessing
from eclipsetools.alignment import find_translation, find_transform
from eclipsetools.utils.image_reader import open_image

# Generate a fixed test seed to ensure reproducible tests
TEST_SEED = 122807528840384100672342137672332424406


def test_align_parametrized(align_params):
    """Individual test case for alignment with specific parameters."""
    ref_image = open_image(r'tests\images\eclipse_5ms.CR3')
    offset, rotation, scale = align_params

    # No noise for this test, same as in the original test
    noise_image = np.zeros_like(ref_image, dtype=np.float32)

    # Call the test function directly without joblib parallelization
    error = _find_transform_error(ref_image, offset, rotation, scale, noise_image)

    scale_error, rotation_error, translation_error = error
    assert scale_error < 0.02, f'Scale error too high: {scale_error}'
    assert rotation_error < 0.3, f'Rotation error too high: {rotation_error}'
    assert translation_error < 1.0, f'Translation error too high: {translation_error}'


def test_translate_parametrized(translate_params):
    """Individual test case for translation with specific parameters."""
    ref_image = open_image(r'tests\images\eclipse_5ms.CR3')
    offset = translate_params

    # No noise for this test, same as in the original test
    noise_image = np.zeros_like(ref_image, dtype=np.float32)

    # Call the test function directly without joblib parallelization
    error = _find_test_image_translation(ref_image, preprocessing.preprocess_for_alignment(ref_image, 1.2, 2.0), offset,
                                         noise_image)

    assert error is not None and error < 0.2, f'Translation error too high for offset {offset}: {error}'


def pytest_generate_tests(metafunc):
    """Generate test cases for test_align_parametrized and test_translate_parametrized functions."""
    # Generate parameters for test_align_parametrized
    if "align_params" in metafunc.fixturenames:
        # Use the same random number generator as in the original test
        rng = np.random.default_rng(TEST_SEED)
        num_tests = 30

        # Generate the same test parameters as in the original test
        offsets = rng.uniform(-20.0, 20.0, (num_tests, 2))
        rotations = rng.uniform(-85.0, 85.0, num_tests)
        scales = rng.uniform(0.8, 1.2, num_tests)

        # Create a list of test cases with IDs
        test_cases = []
        ids = []

        for i in range(num_tests):
            test_cases.append((offsets[i], rotations[i], scales[i]))
            ids.append(
                f"offset=({offsets[i][0]:.2f}, {offsets[i][1]:.2f}) rot={rotations[i]:.2f} scale={scales[i]:.2f}")

        metafunc.parametrize("align_params", test_cases, ids=ids)

    # Generate parameters for test_translate_parametrized
    if "translate_params" in metafunc.fixturenames:
        rng = np.random.default_rng(TEST_SEED)
        num_tests = 30

        # Generate the same offsets as in the original test_translate
        offsets = rng.uniform(-40.0, 40.0, (num_tests, 2))

        # Create a list of test cases with IDs
        test_cases = []
        ids = []

        for i in range(num_tests):
            test_cases.append(offsets[i])
            ids.append(f"offset=({offsets[i][0]:.1f}, {offsets[i][1]:.1f})")

        metafunc.parametrize("translate_params", test_cases, ids=ids)


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
    translated_test_image = preprocessing.preprocess_for_alignment(noisy_test_image, 1.2, 2.0)
    found_translation = find_translation(ref_image_preproc, translated_test_image, low_pass_sigma=0.2)
    error = np.sqrt(np.sum(np.square(found_translation - offset)))
    return error


def _find_transform_error(ref_image: np.ndarray,
                          offset: np.ndarray,
                          rotation: float,
                          scale: float,
                          noise_image: np.ndarray) -> tuple[float, float, float]:
    # We crop images to avoid artifacts at the edges that can occur due to the affine transformation
    crop_margin = 500
    # Cropping has to be done before preprocessing to avoid artifacts at the edges
    ref_image_preproc = preprocessing.preprocess_for_alignment(
        ref_image[crop_margin:-crop_margin, crop_margin:-crop_margin, :], 1.2, 2.0)

    test_image = _transform_image(ref_image, offset, rotation, scale)
    noisy_test_image = np.clip(test_image + noise_image, 0.0, 1.0, dtype=np.float32)

    preproc_image = preprocessing.preprocess_for_alignment(
        noisy_test_image[crop_margin:-crop_margin, crop_margin:-crop_margin, :], 1.2, 2.0)

    recovered_scale, recovered_rotation, recovered_translation = find_transform(ref_image_preproc, preproc_image, 0.2)
    scale_error = float(abs(1.0 - recovered_scale / scale))
    rotation_error = float(abs(rotation - recovered_rotation))
    translation_error = float(np.sqrt(np.sum(np.square(recovered_translation - offset))))
    return scale_error, rotation_error, translation_error


def _transform_image(image, translation, rotation, scale):
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
