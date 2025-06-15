import numpy as np
import rawpy


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
