import fractions
import json
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
import rawpy


@dataclass
class Image:
    data: np.ndarray
    exposure_ms: float | None


def open_raw_image(path: str) -> Image:
    exposure_ms = _get_exposure_ms(path)
    with rawpy.imread(path) as raw:
        data = np.float32(
            raw.postprocess(
                output_bps=16,
                user_flip=0,
                gamma=(1.0, 1.0),
                user_wb=[1.0, 1.0, 1.0, 1.0],
                output_color=rawpy.ColorSpace.raw,
                no_auto_bright=True)) / 65535.0
    return Image(data=data, exposure_ms=exposure_ms)


def _get_exposure_ms(path: str) -> float:
    if shutil.which("exiftool/exiftool") is None:
        raise RuntimeError("ExifTool is not installed or not in PATH.")
    result = subprocess.run(
        ["exiftool/exiftool", "-j", "-ExposureTime", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True
    )
    metadata = json.loads(result.stdout)
    exposure_string = metadata[0].get("ExposureTime")
    exposure_ms = 1000.0 * float(fractions.Fraction(str(exposure_string)))
    return exposure_ms
