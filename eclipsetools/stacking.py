import numpy as np
from sklearn.linear_model import RANSACRegressor


def linear_fit(x, y, max_points=10000):
    # We care more about the top part of the signal than the noisy bottom part, which can skew the linear fit.
    # We only use about max_points data points to find the percentile, because that is slow.
    percentile_limiter = max(len(x) // max_points, 1)
    lower_limit_x = np.percentile(x[::percentile_limiter], 80)
    lower_limit_y = np.percentile(y[::percentile_limiter], 80)
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

    model = RANSACRegressor()
    model.fit(x_valid.reshape(-1, 1), y_valid)
    linear_coef = model.estimator_.coef_[0]
    linear_intercept = model.estimator_.intercept_
    return linear_coef, linear_intercept


def weight_function_sigmoid(arr):
    return np.minimum(
        _sigmoid_weight(arr, 0.09, 0.012), 1 - _sigmoid_weight(arr, 0.75, 0.03)
    )


def _sigmoid_weight(x, center=0.5, width=0.1):
    return 1 / (1 + np.exp(-(x - center) / width))
