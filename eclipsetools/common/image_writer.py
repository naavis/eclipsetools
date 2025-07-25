import numpy as np
from tifffile import tifffile
from PIL import ImageCms


def save_tiff(image: np.ndarray, output_path: str, embed_srgb: bool = False):
    """
    Save image to the specified output path as tiff
    :param image: Image to save
    :param output_path: Path where the image will be saved
    :param embed_srgb: Whether to embed sRGB ICC profile
    :return: None
    """
    if embed_srgb:
        # Get standard sRGB profile
        srgb_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        icc_profile = srgb_profile.tobytes()

        tifffile.imwrite(
            output_path,
            image,
            compression="zlib",
            extratags=[(34675, "B", len(icc_profile), icc_profile, False)],
        )
    else:
        tifffile.imwrite(output_path, image, compression="zlib")
