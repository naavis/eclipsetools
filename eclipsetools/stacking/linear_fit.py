from typing import Any

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from sklearn.linear_model import RANSACRegressor, LinearRegression

from eclipsetools.utils.circle_finder import find_circle, get_binary_moon_mask
from eclipsetools.utils.image_reader import open_image


def fit_eclipse_image_pair(
    image_path_a: str,
    image_path_b: str,
    fit_intercept: bool,
    moon_mask_size: float,
    moon_min_radius: int = 400,
    moon_max_radius: int = 600,
) -> tuple[str, str, tuple[float, float]]:
    """
    Fit linear relationship between two eclipse images.

    :param image_path_a: Path to first image
    :param image_path_b: Path to second image
    :param fit_intercept: Whether to fit the intercept in the linear regression
    :param moon_mask_size: Size of the moon mask relative to the moon radius
    :param moon_min_radius: Minimum radius of the moon to detect in pixels
    :param moon_max_radius: Maximum radius of the moon to detect in pixels
    :return: Tuple of (image_path_a, image_path_b, (linear_coef, linear_intercept))
    """
    img_a = open_image(image_path_a)
    img_b = open_image(image_path_b)

    moon_img_a = find_circle(
        img_a[:, :, 1], min_radius=moon_min_radius, max_radius=moon_max_radius
    )
    moon_mask_a = get_binary_moon_mask(img_a.shape, moon_img_a, moon_mask_size)

    moon_img_b = find_circle(
        img_b[:, :, 1], min_radius=moon_min_radius, max_radius=moon_max_radius
    )
    moon_mask_b = get_binary_moon_mask(img_b.shape, moon_img_b, moon_mask_size)

    # We linear fit only pixels that are not contaminated by the moon in either image.
    pixels_without_moon = moon_mask_a & moon_mask_b
    img_a_points = img_a[pixels_without_moon, :].ravel()
    img_b_points = img_b[pixels_without_moon, :].ravel()

    linear_coef, linear_intercept = _linear_fit(
        img_a_points, img_b_points, fit_intercept
    )

    return image_path_a, image_path_b, (linear_coef, linear_intercept)


def _linear_fit(x, y, fit_intercept, max_points=10000):
    # We care more about the top part of the signal than the noisy bottom part, which can skew the linear fit.
    # We only use about max_points data points to find the percentile, because that is slow.
    percentile_limiter = max(len(x) // max_points, 1)
    lower_limit_x = min(0.2, np.percentile(x[::percentile_limiter], 90))
    lower_limit_y = min(0.2, np.percentile(y[::percentile_limiter], 90))
    upper_limit = 0.9

    valid_points = (
        (x > lower_limit_x)
        & (x < upper_limit)
        & (y > lower_limit_y)
        & (y < upper_limit)
    )

    x_valid = x[valid_points]
    y_valid = y[valid_points]

    # Only use about max_points data points for the linear fit to avoid performance issues
    div = max(len(x_valid) // max_points, 1)
    x_valid = x_valid[::div]
    y_valid = y_valid[::div]

    model = RANSACRegressor(
        min_samples=100, estimator=LinearRegression(fit_intercept=fit_intercept)
    )
    model.fit(x_valid.reshape(-1, 1), y_valid)
    linear_coef = model.estimator_.coef_[0]
    linear_intercept = model.estimator_.intercept_

    return linear_coef, linear_intercept


def solve_global_linear_fits(
    pairwise_fits: list[tuple[str, str, tuple[float, float]]], ref_image_path: str
) -> dict[str, tuple[Any, Any]]:
    """
    Solve global linear fits for multiple image pairs using least squares.
    This function builds a system of equations based on the pairwise linear fits and solves it to find the best linear
    coefficients for each image.

    Each image I_k can be mapped to the reference image space using linear coefficients a_k and b_k by the equation
    I_ref(p) = a_k * I_k(p) + b_k, where I_ref is the pixel value in the reference image space for pixel p.

    We have pairwise fits between images I_i and I_j: I_j = a_ij * I_i + b_ij

    Image j is related to the reference space with a_j * I_j + b_j, and we want to find a_j and b_j.
    By substituting the pairwise fit into the equation, we get a_j * I_j + b_j = (a_j * a_ij) * I_i + (a_j * b_ij + b_j)

    Both images describe the same scene radiance, so we want a_i * I_i + b_i = (a_j * a_ij) * I_i + (a_j * b_ij + b_j)

    This leads to the constraints a_i = a_j * a_ij and b_i = a_j * b_ij + b_j, or with some rearranging:
    a_j - a_ij * a_i = 0
    b_j - a_ij * b_i = b_ij.

    With these constraints and knowing that a_ref = 1 and b_ref = 0, we can solve for all a_k and b_k using least
    squares.

    :param pairwise_fits: List of (img_a_path, img_b_path, (a_ij, b_ij)) from linear fits
    :param ref_image_path: Path to the reference image, which must be one of the images in pairwise_fits
    :return: Dictionary mapping image paths to their linear coefficients (a_k, b_k)
    """
    # Build map from paths to indices
    image_paths = sorted(set([p for a, b, _ in pairwise_fits for p in (a, b)]))
    index_map = {path: i for i, path in enumerate(image_paths)}
    n_images = len(image_paths)

    n_vars = 2 * n_images  # (a_i, b_i) per image
    equations = []
    values = []

    # Add pairwise equations
    for path_i, path_j, (a_ij, b_ij) in pairwise_fits:
        i = index_map[path_i]
        j = index_map[path_j]

        # a_j - a_ij * a_i = 0
        row_a = np.zeros(n_vars)
        row_a[2 * j] = 1  # a_j
        row_a[2 * i] = -a_ij  # -a_ij * a_i
        equations.append(row_a)
        values.append(0.0)

        # b_j - a_ij * b_i - b_ij = 0
        row_b = np.zeros(n_vars)
        row_b[2 * j + 1] = 1  # b_j
        row_b[2 * i + 1] = -a_ij  # -a_ij * b_i
        equations.append(row_b)
        values.append(b_ij)

    # Add fixed constraint: a_0 = 1, b_0 = 0
    ref_image_index = index_map[ref_image_path]

    row_a_ref = np.zeros(n_vars)
    row_a_ref[2 * ref_image_index] = 1
    equations.append(row_a_ref)
    values.append(1.0)

    row_b_ref = np.zeros(n_vars)
    row_b_ref[2 * ref_image_index + 1] = 1
    equations.append(row_b_ref)
    values.append(0.0)

    # Build sparse matrix and solve
    coef_matrix = lil_matrix((len(equations), n_vars))
    for i, row in enumerate(equations):
        coef_matrix.rows[i] = list(np.nonzero(row)[0])
        coef_matrix.data[i] = list(row[row != 0])

    b = np.array(values)
    result: np.ndarray = lsqr(coef_matrix.tocsr(), b)[
        0
    ]  # result is [a_0, b_0, a_1, b_1, ..., a_N, b_N]

    # Return results as dict
    result_dict = {
        path: (result[2 * i], result[2 * i + 1]) for path, i in index_map.items()
    }

    return result_dict
