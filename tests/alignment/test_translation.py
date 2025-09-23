import cv2
import numpy as np

from eclipsetools.alignment.translation import find_translation
from eclipsetools.common.image_reader import open_image
from eclipsetools.preprocessing import workflows


def test_translate_parametrized(translate_params):
    """Individual test case for translation with specific parameters."""
    ref_image = open_image(r"tests\images\eclipse_5ms.CR3")
    offset = translate_params

    # Call the test function directly without joblib parallelization
    error = _find_test_image_translation(
        ref_image,
        workflows.preprocess_with_auto_mask(ref_image, 1.2, 2.0, 0, 400, 600),
        offset,
    )

    assert (
        error is not None and error < 0.2
    ), f"Translation error too high for offset {offset}: {error}"


def _find_test_image_translation(
    ref_image: np.ndarray,
    ref_image_preproc: np.ndarray,
    offset: np.ndarray,
) -> float:
    offset_y, offset_x = offset
    translation_matrix = np.array(
        [[1, 0, offset_x], [0, 1, offset_y]], dtype=np.float32
    )
    test_image = cv2.warpAffine(
        ref_image, translation_matrix, dsize=(ref_image.shape[1], ref_image.shape[0])
    )
    translated_test_image = workflows.preprocess_with_auto_mask(
        test_image, 1.2, 2.0, 0, 400, 600
    )
    found_translation = find_translation(
        ref_image_preproc, translated_test_image, low_pass_sigma=0.2
    )
    error = np.sqrt(np.sum(np.square(found_translation - offset)))
    return error
