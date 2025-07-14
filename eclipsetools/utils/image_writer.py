import numpy as np
from tifffile import tifffile


def save_tiff(image: np.ndarray, output_path: str):
    """
    Save image to the specified output path as tiff
    :param image: Image to save
    :param output_path: Path where the image will be saved
    :return: None
    """
    tifffile.imwrite(output_path, image, compression='zlib')
