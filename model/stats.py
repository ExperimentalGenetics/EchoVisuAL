import numpy as np


def calculate_entropy_for_2_classes(class1, class2):
    """
    Method for the calculation of the entropy for two classes
    :param class1: first class probability
    :param class2: second class probability
    :return: entropy of two classes
    """
    class1_clipped = np.clip(class1, 1e-10, 1.0 - 1e-10)
    class2_clipped = np.clip(class2, 1e-10, 1.0 - 1e-10)
    return -((class1_clipped * np.log2(class1_clipped)) + (class2_clipped * np.log2(class2_clipped)))


def calculate_uncertainty_metrics(mean_background, mean_trace, mean_entropy):
    """
    Method for the calculation of uncertainty metrics
    :param mean_background: mean of the background probabilities
    :param mean_trace: mean of the trace probabilities
    :param mean_entropy: mean of the entropy
    :return:
    """
    pixel_wise_entropy = calculate_entropy_for_2_classes(mean_background, mean_trace)
    pixel_wise_bald = np.subtract(pixel_wise_entropy, mean_entropy)

    um = {
        'pixel_wise_entropy': pixel_wise_entropy,
        'pixel_wise_bald': pixel_wise_bald,
        'frame_entropy': np.sum(pixel_wise_entropy),
        'frame_bald': np.sum(pixel_wise_bald)
    }

    return um
