import numpy as np
from sklearn.linear_model import RANSACRegressor


def linear_fit(x, y):
    # TODO: Parametrize upper and lower limits
    lower_limit = 0.005
    upper_limit = 0.9
    valid_points = (
        (x > lower_limit) & (x < upper_limit) & (y > lower_limit) & (y < upper_limit)
    )
    x_valid = x[valid_points]
    y_valid = y[valid_points]
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
