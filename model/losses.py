import tensorflow as tf

from keras import backend as k
from keras.backend import binary_crossentropy


def dice_coefficient(y_true, y_pred, smooth=1e-7):
    """
    Calculation of the dice coefficient
    :param y_true: y_true
    :param y_pred: y_pred
    :param smooth: numerical value for smoothing
    :return: dice score
    """
    intersection = k.sum(k.abs(y_true * y_pred), axis=-1)
    dice = (2.0 * intersection + smooth) / (
            k.sum(k.square(y_true), -1) + k.sum(k.square(y_pred), -1) + smooth)
    return dice


def dice_loss(y_true, y_pred):
    """
    Calculation of the dice loss
    :param y_true: y_true
    :param y_pred: y_pred
    :return: dice_loss
    """
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator


def combined_loss(y_true, y_pred):
    """
    Combined loss function
    :param y_true: y_true
    :param y_pred: y_pred
    :return: sum of binary crossentropy and dice loss
    """
    y_true_cast = tf.cast(y_true, tf.float64)
    y_pred_cast = tf.cast(y_pred, tf.float64)
    log_loss = binary_crossentropy(y_true_cast, y_pred_cast)
    dice = dice_loss(y_true_cast, y_pred_cast)
    return log_loss + dice
