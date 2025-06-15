from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectedCircle:
    center: tuple
    radius: float


def find_circle(
        image: np.ndarray,
        min_radius: int,
        max_radius: int) -> DetectedCircle | None:
    assert image.ndim == 2, "Input image must be grayscale (2D array)."

    detected_circles = cv2.HoughCircles(
        image=(image * 255).astype(np.uint8),
        method=cv2.HOUGH_GRADIENT,
        dp=3,  # Inverse of accumulator resolution, i.e. 3 means 1/3 resolution of original image
        minDist=image.shape[0] / 16.0,  # Minimum distance between found circles
        param1=100,  # Upper threshold for Canny edge detector
        param2=10,  # Accumulator threshold for finding images (smaller -> more circles detected)
        minRadius=min_radius,
        maxRadius=max_radius)

    if detected_circles is not None:
        # plot_circles(detected_circles, image)

        circle = detected_circles[0][0]
        return DetectedCircle(center=(circle[1], circle[0]), radius=float(circle[2]))
    else:
        return None


def plot_circles(detected_circles, image):
    print(f'Found {detected_circles.shape[1]} circles')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        print(f'Circle at ({a}, {b}) with radius {r}')

        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        ax.imshow(image)
        ax.add_patch(Circle((a, b), r, fill=False, edgecolor='red'))
        plt.show()
