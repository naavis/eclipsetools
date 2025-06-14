import numpy as np
import cv2

class DetectedCircle:
    def __init__(self, x, y, radius):
        self.center = (x, y)
        self.radius = radius


def find_circle(
        image: np.ndarray,
        min_radius: int,
        max_radius: int) -> DetectedCircle | None:
    assert image.ndim == 2, "Input image must be grayscale (2D array)."

    detected_circles = cv2.HoughCircles(
        image=(image * 255).astype(np.uint8),
        method=cv2.HOUGH_GRADIENT,
        dp=3, # Inverse of accumulator resolution, i.e. 3 means 1/3 resolution of original image
        minDist=image.shape[0] / 16.0, # Minimum distance between found circles
        param1=200, # Upper threshold for Canny edge detector
        param2=100, # Accumulator threshold for finding images (smaller -> more circles detected)
        minRadius=min_radius,
        maxRadius=max_radius)

    if detected_circles is not None:
        circle = detected_circles[0][0]
        return DetectedCircle(circle[1], circle[0], circle[2])
    else:
        return None
