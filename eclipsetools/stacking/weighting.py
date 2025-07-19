import numpy as np


def weight_function_sigmoid(arr, clip_lower=True, clip_upper=True):
    lower = _sigmoid_weight(arr, 0.1, 0.01) if clip_lower else np.ones_like(arr)
    upper = 1 - _sigmoid_weight(arr, 0.75, 0.03) if clip_upper else np.ones_like(arr)
    return np.minimum(lower, upper)


def _sigmoid_weight(x, center=0.5, width=0.1):
    return 1 / (1 + np.exp(-(x - center) / width))
