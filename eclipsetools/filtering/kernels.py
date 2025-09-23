import numba
import numpy as np


def get_kernel_size(sigma_tangent: float, sigma_radial: float) -> int:
    """
    Calculate the kernel size based on the maximum of the two sigmas.
    The kernel size is always odd and at least 3.
    """
    max_sigma = max(sigma_tangent, sigma_radial)
    kernel_size = int(max_sigma * 4) | 1  # Ensure it's odd
    return max(kernel_size, 3)  # Ensure minimum size of 3


@numba.jit(nogil=True)
def adaptive_kernel(
    r_grid: np.ndarray,
    theta_grid: np.ndarray,
    sigma_tangent: float,
    sigma_radial: float,
) -> np.ndarray:
    """
    Create an adaptive kernel based on the distance from the center and angle.
    The kernel size is deduced from r_grid size.
    """
    cx = r_grid.shape[1] // 2
    cy = r_grid.shape[0] // 2
    cr = r_grid[cy, cx]  # Center radius
    ct = theta_grid[cy, cx]  # Center angle

    # Calculate circular angular distance (handles 0/2π wraparound)
    theta_diff = theta_grid - ct
    theta_diff = np.minimum(np.abs(theta_diff), 2 * np.pi - np.abs(theta_diff))

    exponent = np.zeros_like(r_grid, dtype=np.float32)
    # Tiny sigma values can cause artifacts, and an unsharp mask of less than 0.1 does not make any practical sense
    if sigma_radial > 0.1:
        exponent += -((r_grid - cr) ** 2) / (2 * sigma_radial**2)
    if sigma_tangent > 0.1:
        exponent += -((r_grid * theta_diff) ** 2) / (2 * sigma_tangent**2)

    return np.exp(exponent).astype(np.float32)
