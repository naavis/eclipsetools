from enum import StrEnum

import cv2
import numpy as np

from eclipsetools.utils.circle_finder import DetectedCircle, find_circle
from eclipsetools.utils.image_reader import open_image


class MaskMode(StrEnum):
    AUTO_PER_IMAGE = "auto"
    MAXIMUM = "max"


def annulus_mask(
    shape: tuple,
    moon_center: tuple,
    mask_inner_radius_px: float,
    mask_outer_radius_px: float | None,
) -> np.ndarray:
    if mask_outer_radius_px is None or mask_outer_radius_px < 0.0:
        mask = _circle_mask_internal(shape, moon_center, mask_inner_radius_px)
    else:
        mask = _annulus_mask_internal(
            shape, moon_center, mask_inner_radius_px, mask_outer_radius_px
        )
    return mask


def hann_window_mask(shape: tuple) -> np.ndarray:
    assert len(shape) == 2, "Shape must be a 2D tuple (height, width)."
    return cv2.createHanningWindow(shape[::-1], cv2.CV_32F)


def _annulus_mask_internal(
    shape: tuple, mask_center: tuple, inner_radius: float, outer_radius: float
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)

    cv2.circle(
        img=mask,
        center=(int(mask_center[1]), int(mask_center[0])),
        radius=int(outer_radius),
        color=(1.0,),
        thickness=-1,
    )
    cv2.circle(
        img=mask,
        center=(int(mask_center[1]), int(mask_center[0])),
        radius=int(inner_radius),
        color=(0.0,),
        thickness=-1,
    )
    cv2.GaussianBlur(src=mask, ksize=(0, 0), sigmaX=20, sigmaY=20, dst=mask)

    return mask


def _circle_mask_internal(
    shape: tuple, mask_center: tuple, radius: float
) -> np.ndarray:
    mask = np.ones(shape, dtype=np.float32)

    cv2.circle(
        img=mask,
        center=(int(mask_center[1]), int(mask_center[0])),
        radius=int(radius),
        color=(0.0,),
        thickness=-1,
    )

    cv2.GaussianBlur(src=mask, ksize=(0, 0), sigmaX=20, sigmaY=20, dst=mask)

    return mask


def find_mask_radii_px(
    image: np.ndarray,
    moon: DetectedCircle,
    mask_inner_radius_multiplier: float,
    mask_outer_radius_multiplier: float,
) -> tuple[float, float | None]:
    saturated_radius = _estimate_saturated_radius(moon, image)
    mask_inner_radius_px = mask_inner_radius_multiplier * (
        saturated_radius if saturated_radius else moon.radius
    )

    shortest_distance_to_edge = min(
        moon.center[0],
        moon.center[1],
        image.shape[0] - moon.center[0],
        image.shape[1] - moon.center[1],
    )

    # We limit the outer radius of the annulus mask to avoid intersecting the image edges.
    mask_outer_radius_px = (
        min(
            mask_outer_radius_multiplier * mask_inner_radius_px,
            shortest_distance_to_edge * 0.95,
        )
        if mask_outer_radius_multiplier > 0.0
        else None
    )

    return mask_inner_radius_px, mask_outer_radius_px


def find_mask_inner_radius_px(
    image_path: str, inner_multiplier: float, moon_min_radius: int, moon_max_radius: int
) -> float:
    rgb_image = open_image(image_path)
    image = np.mean(rgb_image, axis=2, dtype=np.float32)
    moon = find_circle(image, moon_min_radius, moon_max_radius)
    saturated_radius = _estimate_saturated_radius(moon, rgb_image)
    inner_radius_px = inner_multiplier * (
        saturated_radius if saturated_radius else moon.radius
    )
    return inner_radius_px


def _estimate_saturated_radius(
    moon_params: DetectedCircle, raw_image: np.ndarray
) -> float | None:
    saturated_pixels = (
        cv2.erode(
            np.max(raw_image, axis=2), kernel=np.ones((3, 3), np.float32), iterations=1
        )
        > 0.999
    )
    saturated_radius = None
    if np.any(saturated_pixels):
        distances = np.linalg.norm(
            np.argwhere(saturated_pixels) - moon_params.center, axis=1
        )
        saturated_radius = np.max(distances)
    return saturated_radius
