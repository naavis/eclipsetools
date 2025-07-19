def weight_function_hat(image):
    """
    Weight function for stacking images, using a quadratic "hat" function. The smoothly goes from (0.0, 0.0) to
    (0.5, 1.0) and back to (1.0, 0.0).
    :param image: Image data as a Numpy array, normalized to [0.0, 1.0].
    :return: Weights as a Numpy array, same shape as the input image.
    """
    return 4.0 * image * (1.0 - image)
