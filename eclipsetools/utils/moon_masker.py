import cv2
import numpy as np

from eclipsetools.utils.circle_finder import find_circle


def get_precise_moon_mask(
    image: np.ndarray, min_moon_radius: int, max_moon_radius: int
):
    """
    Create a precise mask of the moon limb in the image using polar coordinates.
    This function finds the moon in the image, transforms the image to polar coordinates,
    estimates the brightness profile of the moon limb, and creates a mask based on the brightness profile.
    :param min_moon_radius: Minimum radius of the moon in pixels for moon detection.
    :param max_moon_radius: Maximum radius of the moon in pixels for moon detection.
    :param image: Image in which to find the moon limb.
    :return: A mask where the moon is represented by 0.0 and the rest of the image by 1.0.
    """
    circle = find_circle(image.mean(axis=2), min_moon_radius, max_moon_radius)

    # Lower and upper limits for the polar transformation radius
    polar_min_radius = 0.95
    polar_max_radius = 1.05
    polar_img = cv2.warpPolar(
        image,
        (image.shape[1], image.shape[0]),
        (circle.center[1], circle.center[0]),
        circle.radius * polar_max_radius,
        cv2.WARP_POLAR_LINEAR | cv2.INTER_LANCZOS4,
    )

    start_index = int(polar_min_radius * polar_img.shape[1] / polar_max_radius)
    moon_limb_area = polar_img[:, start_index:]

    # Calculate limb brightness profile across +- half_width pixels from edge_index
    # TODO: Make sure the half_width is not larger than the image width
    limb_profile = _estimate_limb_profile(moon_limb_area, half_width=100)

    # We define the edge of the moon limb as the maximum gradient in the moon limb area
    edge_indices = np.argmax(np.gradient(moon_limb_area, axis=1), axis=1)
    edge_indices = np.median(edge_indices, axis=1).astype(
        np.int64
    )  # We pick the middle value of each RGB channel

    # We find the index of the maximum gradient in the limb profile to match it with edge_index
    limb_profile_edge_index = np.argmax(np.gradient(limb_profile))

    polar_mask = np.zeros(polar_img.shape[:2], dtype=np.float32)
    for i, edge_index in enumerate(edge_indices):
        # edge_index is referenced to the cropped polar image, and we want to map it to the full polar image
        start_i = start_index + edge_index - limb_profile_edge_index
        end_i = start_i + limb_profile.shape[0]
        # The mask is inverted here, because OpenCV fills areas outside the polar image with 0.0 when converting to
        # cartesian coordinates. By inverting the mask here and then inverting the cartesian mask again, we don't have
        # to deal with that.
        polar_mask[i, start_i:end_i] = 1.0 - limb_profile
        polar_mask[i, :start_i] = 1.0

    cartesian_mask = 1.0 - cv2.warpPolar(
        polar_mask,
        (image.shape[1], image.shape[0]),
        (circle.center[1], circle.center[0]),
        circle.radius * polar_max_radius,
        cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )

    return cartesian_mask


def _estimate_limb_profile(moon_limb_polar: np.ndarray, half_width: int) -> np.ndarray:
    gradient = np.gradient(moon_limb_polar, axis=1)
    edge_index = np.argmax(gradient, axis=1)

    rows, cols, channels = moon_limb_polar.shape
    row_indices = np.arange(rows)[:, None, None]  # Shape: (rows, 1, 1)
    channel_indices = np.arange(channels)[None, None, :]  # Shape: (1, 1, channels)

    # Create column indices for given number of pixels around edge_index
    col_offsets = np.arange(-half_width, half_width)  # Shape: (half_width * 2,)
    col_indices = (
        edge_index[:, None, :] + col_offsets[None, :, None]
    )  # Shape: (rows, half_width * 2, channels)

    # Clip indices to valid range
    col_indices = np.clip(col_indices, 0, cols - 1)

    # Extract the pixels belonging to range
    selected_pixels = moon_limb_polar[row_indices, col_indices, channel_indices]

    # Shape: (rows, half_width * 2, channels)
    selected_pixels -= np.min(selected_pixels, axis=1, keepdims=True)
    selected_pixels /= np.max(selected_pixels, axis=1, keepdims=True)

    # Here we calculate the mean profile across all rows of the polar coordinate image, and all color channels
    mean_profile = np.mean(selected_pixels, axis=(0, 2))  # Length half_width * 2
    mean_profile -= mean_profile.min()
    mean_profile /= mean_profile.max()

    # We find the half maximum point of the profile, and the real maximum point
    max_index = mean_profile.argmax()
    middle_index = np.argmin(np.abs(mean_profile - 0.5))

    # The bottom portion of the profile might be contaminated by refracted light, Earthshine, and glare
    # We assume the brightness profile is symmetric, so we can use the top half to approximate the brightness profile
    top_portion = mean_profile[middle_index:max_index]
    mirrored_top_portion = -top_portion[::-1] + 1
    full_profile = np.concatenate((mirrored_top_portion, top_portion[1:]))
    full_profile -= full_profile.min()
    full_profile /= full_profile.max()

    return full_profile
