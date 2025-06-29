import os

import cv2
import numpy as np
import rawpy


def open_image(path: str) -> np.ndarray:
    """
    Open either a raw image file or a TIFF file and return as normalized float32 array.
    :param path: Path to the image file
    :return: Normalized image array as float32 with values in range [0, 1]
    """
    _, ext = os.path.splitext(path.lower())

    if ext in ['.tiff', '.tif']:
        # Read TIFF file
        image = cv2.cvtColor(cv2.imread(path, flags=cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

        # Convert to float32 and normalize
        if image.dtype == np.uint8:
            return np.float32(image) / 255.0
        elif image.dtype == np.uint16:
            return np.float32(image) / 65535.0
        elif image.dtype in [np.float32, np.float64]:
            # Assume already normalized or handle appropriately
            return np.float32(image)
        else:
            # For other dtypes, convert to float32 and normalize by max value
            return np.float32(image) / np.iinfo(image.dtype).max
    else:
        # Assume it's a raw file and use the existing function
        return _open_raw_image(path)


def _open_raw_image(path: str) -> np.ndarray:
    """
    Open a raw image file using rawpy and return as normalized float32 array.
    :param path: Path to the raw image file
    :return: Normalized image array as float32 with values in range [0, 1]
    """
    with rawpy.imread(path) as raw:
        return np.float32(
            raw.postprocess(
                output_bps=16,
                user_flip=0,
                gamma=(1.0, 1.0),
                user_wb=[1.0, 1.0, 1.0, 1.0],
                output_color=rawpy.ColorSpace.raw,
                no_auto_bright=True)) / 65535.0
