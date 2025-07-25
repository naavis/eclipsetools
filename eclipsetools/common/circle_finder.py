from dataclasses import dataclass

import cv2
import numpy as np
import skimage.feature
from scipy.optimize import least_squares


@dataclass
class DetectedCircle:
    center: tuple[float, float]
    radius: float


def find_circle(
    image: np.ndarray, min_radius: int, max_radius: int
) -> DetectedCircle | None:
    assert image.ndim == 2, "Input image must be grayscale (2D array)."

    processed_image = ((np.clip(image, 0.0, 1.0) ** 0.5) * 255).astype(np.uint8)

    detected_circles = cv2.HoughCircles(
        image=processed_image,
        method=cv2.HOUGH_GRADIENT,
        dp=3,  # Inverse of accumulator resolution, i.e. 3 means 1/3 resolution of original image
        minDist=image.shape[0] / 16.0,  # Minimum distance between found circles
        param1=50,  # Upper threshold for Canny edge detector
        param2=30,  # Accumulator threshold for finding images (smaller -> more circles detected)
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if detected_circles is not None:
        # _plot_circles(detected_circles, image)
        x, y, r = detected_circles[0][0]
        circle_object = DetectedCircle(center=(float(y), float(x)), radius=float(r))
        refined = _refine_circle(processed_image, circle_object)
        return refined
    else:
        return None


def _refine_circle(image: np.ndarray, approx_circle: DetectedCircle):
    y0, x0 = map(int, approx_circle.center)
    r0 = approx_circle.radius
    r_crop = int(r0 * 1.3)  # Increase radius for cropping

    y_start = max(0, y0 - r_crop)
    y_end = min(image.shape[0], y0 + r_crop)
    x_start = max(0, x0 - r_crop)
    x_end = min(image.shape[1], x0 + r_crop)

    crop = image[y_start:y_end, x_start:x_end]

    edges = skimage.feature.canny(crop, sigma=2.0)

    y_indices, x_indices = np.nonzero(edges)
    x_indices = x_indices + x_start
    y_indices = y_indices + y_start

    # Filter edge points: only keep those reasonably close to expected circle
    distances = np.sqrt((x_indices - x0) ** 2 + (y_indices - y0) ** 2)
    tolerance = r0 * 0.3  # Only keep points within 30% of expected radius
    mask = np.abs(distances - r0) < tolerance
    x_indices = x_indices[mask]
    y_indices = y_indices[mask]

    def residuals(p):
        x0_fit, y0_fit, r_fit = p
        return np.sqrt((x_indices - x0_fit) ** 2 + (y_indices - y0_fit) ** 2) - r_fit

    # Add bounds to prevent drift
    bounds = (
        [x0 - r0 * 0.2, y0 - r0 * 0.2, r0 * 0.7],
        [x0 + r0 * 0.2, y0 + r0 * 0.2, r0 * 1.3],
    )

    result = least_squares(
        residuals, x0=[x0, y0, r0], bounds=bounds, loss="soft_l1"  # Robust to outliers
    )

    if not result.success:
        return approx_circle

    xc, yc, rc = result.x
    return DetectedCircle(center=(yc, xc), radius=rc)


def _plot_circles(detected_circles, image):
    print(f"Found {detected_circles.shape[1]} circles")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        print(f"Circle at ({a}, {b}) with radius {r}")

        fig, ax = plt.subplots(1)
        ax.set_aspect("equal")
        ax.imshow(image)
        ax.add_patch(Circle((a, b), r, fill=False, edgecolor="red"))
        plt.show()


def get_binary_moon_mask(
    shape: tuple, moon_params: DetectedCircle, mask_size: float
) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    distances = np.sqrt(
        (x - moon_params.center[1]) ** 2 + (y - moon_params.center[0]) ** 2
    )
    moon_mask = distances >= mask_size * moon_params.radius
    return moon_mask
