"""Different losses that can be used as functions in TensorFlow.
"""
from typing import Callable

import tensorflow as tf

from . import metric as Metric

# pylint: disable=missing-function-docstring


def get_loss(loss_name: str) -> Callable:
    """
    Returns loss depending on loss.

    loss should be in {'DICE', 'TVE', 'GDL', 'CEL', 'WCEL'}.

    Returns
    -------
    Callable
        The loss as tensorflow function
    """
    # many returns do not affect the readability
    # pylint: disable=too-many-return-statements

    if loss_name == "DICE":
        return dice_loss

    if loss_name == "DICE-FNR":
        return dice_with_fnr_loss

    if loss_name == "TVE":
        return tversky_loss

    if loss_name == "GDL":
        return generalized_dice_loss

    if loss_name == "GDL-FPNR":
        return generalized_dice_with_fpr_fnr_loss

    if loss_name == "NDL":
        return normalized_dice_loss

    if loss_name == "EDL":
        return equalized_dice_loss

    if loss_name == "CEL":
        return categorical_cross_entropy_loss

    if loss_name == "BCEL":
        return binary_cross_entropy_loss

    if loss_name == "ECEL":
        return equalized_categorical_cross_entropy

    if loss_name == "NCEL":
        return normalized_categorical_cross_entropy

    if loss_name == "WCEL":
        return weighted_categorical_cross_entropy

    if loss_name == "ECEL-FNR":
        return equalized_categorical_cross_entropy_with_fnr

    if loss_name == "WCEL-FPR":
        return weighted_categorical_crossentropy_with_fpr_loss

    if loss_name == "GCEL":
        return generalized_categorical_cross_entropy

    if loss_name == "CEL+DICE":
        return categorical_cross_entropy_and_dice_loss

    if loss_name == "MSE":
        return mean_squared_error_loss

    raise ValueError(loss_name, "is not a supported loss function.")


def categorical_cross_entropy_and_dice_loss(y_true, y_pred):
    loss = 10 * categorical_cross_entropy_loss(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def categorical_cross_entropy_loss(y_true, y_pred):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(loss, name="crossentropy")


def binary_cross_entropy_loss(y_true, y_pred):
    loss = tf.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(loss, name="crossentropy")


def mean_squared_error_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError(name="mean_squared_error")
    return mse(y_true, y_pred)


def mean_squared_error_and_gradient_loss(y_true, y_pred):
    mse = mean_squared_error_loss(y_true, y_pred)
    grad = gradient_loss(y_true, y_pred)
    loss = mse + grad
    return loss


def mean_absolute_error_loss(y_true, y_pred):
    mae = tf.keras.losses.MeanAbsoluteError(name="mean_absolute_error")
    return mae(y_true, y_pred)


def mean_absolute_error_and_gradient_loss(y_true, y_pred):
    mae = mean_absolute_error_loss(y_true, y_pred)
    grad = gradient_loss(y_true, y_pred)
    loss = mae + grad
    return loss


def gradient_loss(y_true, y_pred):
    def gradient(x, axis):
        if axis == 0:
            grad = x[:, 1:] - x[:, :-1]
        elif axis == 1:
            grad = x[:, :, 1:] - x[:, :, :-1]
        elif axis == 2:
            grad = x[:, :, :, 1:] - x[:, :, :, :-1]
        return abs(grad)

    loss = (
        tf.reduce_mean(
            tf.keras.losses.mean_squared_error(gradient(y_true, 0), gradient(y_pred, 0))
        )
        + tf.reduce_mean(
            tf.keras.losses.mean_squared_error(gradient(y_true, 1), gradient(y_pred, 1))
        )
        + tf.reduce_mean(
            tf.keras.losses.mean_squared_error(gradient(y_true, 2), gradient(y_pred, 2))
        )
    )
    return loss


def regression_combi_loss(y_true, y_pred):
    abs_y_true = tf.abs(y_true)
    sample_weight = tf.cast(tf.math.greater(abs_y_true, 0.1), tf.float32) * 0.9 + 0.1
    mae = tf.keras.losses.MeanAbsoluteError(name="combi_loss")
    return mae(y_true, y_pred, sample_weight) + gradient_loss(y_true, y_pred)


def tversky_loss(y_true, y_pred, alpha=0.3, eps=1e-12):
    beta = 1 - alpha
    axis = tf.range(0, tf.rank(y_pred) - 1)
    intersection = tf.reduce_sum(y_pred * y_true, axis=axis)
    f_p = tf.reduce_sum(
        tf.multiply(
            y_pred, tf.cast(tf.math.logical_not(tf.cast(y_true, tf.bool)), tf.float32)
        ),
        axis=axis,
    )
    f_n = tf.reduce_sum(tf.multiply(tf.subtract(1.0, y_pred), y_true), axis=axis)
    quotient = intersection + tf.scalar_mul(alpha, f_p) + tf.scalar_mul(beta, f_n)
    # print('Loss Shapes: ', intersection.shape, quotient)
    loss = (intersection + eps) / (quotient + eps)
    loss = tf.reduce_mean(loss)  # average loss across classes
    return tf.subtract(tf.constant(1, tf.float32), loss, name="tversky")


def dice_loss(y_true, y_pred, eps=1e-12):
    loss = tf.subtract(
        tf.constant(1, tf.float32), Metric.dice_coefficient_tf(y_true, y_pred), name="dice"
    )  # 1-dice to have a loss that can be minimized
    return loss


def soft_dice_loss(y_true, y_pred, smooth=1e-6):
    loss = tf.subtract(
        tf.constant(1, tf.float32),
        Metric.soft_dice_coefficient_tf(y_true, y_pred, smooth=smooth),
        name="dice",
    )  # 1-dice to have a loss that can be minimized
    return loss


def dice_with_fnr_loss(y_true, y_pred, eps=1e-12):
    loss = tf.subtract(
        tf.constant(1, tf.float32), Metric.dice_coefficient_tf(y_true, y_pred), name="dice"
    )  # 1-dice to have a loss that can be minimized
    classes = y_pred.shape[-1]
    y_true_channels = tf.split(y_true, classes, axis=-1)
    y_pred_channels = tf.split(y_pred, classes, axis=-1)
    for channel in [1, 2]:
        gamma = tf.reduce_sum((1 - y_pred_channels[channel]) * y_true_channels[channel]) / (
            tf.reduce_sum(y_true_channels) + eps
        )
        # print('  False Negative Gamma:', gamma)
        loss = loss + gamma
    return loss


def generalized_dice_loss(y_true, y_pred, eps=1e-12):
    axis = tf.range(0, tf.rank(y_pred) - 1)
    weights = tf.divide(1, tf.add(tf.pow(tf.reduce_sum(y_true, axis=axis), 2), eps))
    # print('Weights: ', weights.shape, weights.numpy())
    intersection = tf.reduce_sum(y_pred * y_true, axis=axis)
    # print('Dice Intersection: ', intersection.shape)
    weighted_intersection = tf.reduce_sum(tf.multiply(weights, intersection))
    union = tf.add(tf.reduce_sum(y_pred, axis=axis), tf.reduce_sum(y_true, axis=axis))
    # print('Dice Union: ', union.shape)
    weighted_union = tf.reduce_sum(tf.multiply(weights, union))
    loss = 2 * (weighted_intersection + eps) / (weighted_union + eps)
    loss = tf.subtract(tf.constant(1, tf.float32), loss, name="generalized_dice")
    return loss  # 1-dice to have a loss that can be minimized


def normalized_dice_loss(y_true, y_pred, eps=1e-12):
    axis = tf.range(0, tf.rank(y_pred) - 1)
    n_true = tf.reduce_sum(y_true)
    weights = tf.divide(n_true, tf.add(tf.pow(tf.reduce_sum(y_true, axis=axis), 2), eps))
    # print('Weights: ', weights.shape, weights.numpy())
    intersection = tf.reduce_sum(y_pred * y_true, axis=axis)
    # print('Dice Intersection: ', intersection.shape)
    weighted_intersection = tf.reduce_sum(tf.multiply(weights, intersection))
    union = tf.add(tf.reduce_sum(y_pred, axis=axis), tf.reduce_sum(y_true, axis=axis))
    # print('Dice Union: ', union.shape)
    weighted_union = tf.reduce_sum(tf.multiply(weights, union))
    loss = 2 * (weighted_intersection + eps) / (weighted_union + eps)
    loss = tf.subtract(tf.constant(1, tf.float32), loss, name="generalized_dice")
    return loss  # 1-dice to have a loss that can be minimized


def equalized_dice_loss(y_true, y_pred, eps=1e-12):
    axis = tf.range(0, tf.rank(y_pred) - 1)
    n_true = tf.reduce_sum(y_true)
    weights = tf.divide(n_true - tf.reduce_sum(y_true, axis=axis), tf.add(n_true, eps))
    # print('Weights: ', weights.shape, weights.numpy())
    intersection = tf.reduce_sum(y_pred * y_true, axis=axis)
    # print('Dice Intersection: ', intersection.shape)
    weighted_intersection = tf.reduce_sum(tf.multiply(weights, intersection))
    union = tf.add(tf.reduce_sum(y_pred, axis=axis), tf.reduce_sum(y_true, axis=axis))
    # print('Dice Union: ', union.shape)
    weighted_union = tf.reduce_sum(tf.multiply(weights, union))
    loss = 2 * (weighted_intersection + eps) / (weighted_union + eps)
    loss = tf.subtract(tf.constant(1, tf.float32), loss, name="generalized_dice")
    return loss  # 1-dice to have a loss that can be minimized


def generalized_dice_with_fpr_fnr_loss(y_true, y_pred, eps=1e-12):
    axis = tf.range(0, tf.rank(y_pred) - 1)
    weights = tf.divide(1, tf.add(tf.pow(tf.reduce_sum(y_true, axis=axis), 2), eps))
    # print('Weights: ', weights.shape, weights.numpy())
    intersection = tf.reduce_sum(y_pred * y_true, axis=axis)
    # print('Dice Intersection: ', intersection.shape)
    weighted_intersection = tf.reduce_sum(tf.multiply(weights, intersection))
    union = tf.add(tf.reduce_sum(y_pred, axis=axis), tf.reduce_sum(y_true, axis=axis))
    # print('Dice Union: ', union.shape)
    weighted_union = tf.reduce_sum(tf.multiply(weights, union))
    loss = 2 * (weighted_intersection + eps) / (weighted_union + eps)
    loss = tf.subtract(tf.constant(1, tf.float32), loss, name="generalized_dice")
    # print('  GDL:', loss)
    classes = y_pred.shape[-1]
    y_true_channels = tf.split(y_true, classes, axis=-1)
    y_pred_channels = tf.split(y_pred, classes, axis=-1)
    for channel in [2]:
        gamma = tf.reduce_sum((1 - y_pred_channels[channel]) * y_true_channels[channel]) / (
            tf.reduce_sum(y_true_channels) * 10 + eps
        )
        # print('  False Negative Gamma:', gamma)
        loss = loss + gamma
    for channel in [0]:
        gamma = tf.reduce_sum(y_pred_channels[channel] * (1 - y_true_channels[channel])) / (
            tf.reduce_sum(y_true_channels) * 10 + eps
        )
        # print('  False Positive Gamma:', gamma)
        loss = loss + gamma
    return loss  # 1-dice to have a loss that can be minimized


def normalized_categorical_cross_entropy(y_true, y_pred, eps=1e-12):
    axis = tf.range(0, tf.rank(y_pred) - 1)
    n_true = tf.reduce_sum(y_true)
    weights = tf.divide(n_true, tf.add(tf.pow(tf.reduce_sum(y_true, axis=axis), 2), eps))
    ce_func = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    cat_corr_ent = ce_func(y_true, y_pred)
    classes = y_pred.shape[-1]
    y_true_channels = tf.split(y_true, classes, axis=-1)
    for cls in range(classes):
        per_class_ce = tf.reduce_sum(
            tf.expand_dims(cat_corr_ent, -1) * y_true_channels[cls]
        )
        loss = (
            (per_class_ce * weights[cls])
            if cls == 0
            else loss + (per_class_ce * weights[cls])
        )
    return tf.identity(loss, name="normalized_categorical_cross_entropy")


def equalized_categorical_cross_entropy(y_true, y_pred, eps=1e-12):
    axis = tf.range(0, tf.rank(y_pred) - 1)
    n_true = tf.reduce_sum(y_true)
    weights = tf.divide(n_true - tf.reduce_sum(y_true, axis=axis), tf.add(n_true, eps))
    ce_func = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    cat_corr_ent = ce_func(y_true, y_pred)
    classes = y_pred.shape[-1]
    y_true_channels = tf.split(y_true, classes, axis=-1)
    for cls in range(classes):
        per_class_ce = tf.reduce_sum(
            tf.expand_dims(cat_corr_ent, -1) * y_true_channels[cls]
        )
        loss = (
            (per_class_ce * weights[cls])
            if cls == 0
            else loss + (per_class_ce * weights[cls])
        )
    return tf.identity(loss, name="equalized_categorical_cross_entropy")


def weighted_categorical_cross_entropy(y_true, y_pred, eps=1e-12):
    axis = tf.range(0, tf.rank(y_pred) - 1)
    weights = tf.divide(1, tf.reduce_sum(y_true, axis=axis) + eps)
    ce_func = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    cat_corr_ent = ce_func(y_true, y_pred)
    classes = y_pred.shape[-1]
    y_true_channels = tf.split(y_true, classes, axis=-1)
    for cls in range(classes):
        per_class_ce = tf.reduce_sum(
            tf.expand_dims(cat_corr_ent, -1) * y_true_channels[cls]
        )
        loss = (
            (per_class_ce * weights[cls])
            if cls == 0
            else loss + (per_class_ce * weights[cls])
        )
    return tf.identity(loss, name="weighted_categorical_cross_entropy")


def generalized_categorical_cross_entropy(y_true, y_pred, eps=1e-12):
    axis = tf.range(0, tf.rank(y_pred) - 1)
    weights = tf.divide(1, tf.add(tf.pow(tf.reduce_sum(y_true, axis=axis), 2), eps))
    ce_func = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    cat_corr_ent = ce_func(y_true, y_pred)
    classes = y_pred.shape[-1]
    y_true_channels = tf.split(y_true, classes, axis=-1)
    for cls in range(classes):
        per_class_ce = tf.reduce_sum(
            tf.expand_dims(cat_corr_ent, -1) * y_true_channels[cls]
        )
        loss = (
            (per_class_ce * weights[cls])
            if cls == 0
            else loss + (per_class_ce * weights[cls])
        )
    return tf.identity(loss, name="equalized_categorical_cross_entropy_with_fpr")


def equalized_categorical_cross_entropy_with_fnr(y_true, y_pred, eps=1e-12):
    axis = tf.range(0, tf.rank(y_pred) - 1)
    n_true = tf.reduce_sum(y_true)
    weights = tf.divide(n_true - tf.reduce_sum(y_true, axis=axis), tf.add(n_true, eps))
    ce_func = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    cat_corr_ent = ce_func(y_true, y_pred)
    classes = y_pred.shape[-1]
    y_true_channels = tf.split(y_true, classes, axis=-1)
    y_pred_channels = tf.split(y_pred, classes, axis=-1)
    # print('Weights: ', weights.shape, weights.numpy())
    for cls in range(classes):
        per_class_ce = tf.reduce_sum(
            tf.expand_dims(cat_corr_ent, -1) * y_true_channels[cls]
        )
        loss = (
            (per_class_ce * weights[cls])
            if cls == 0
            else loss + (per_class_ce * weights[cls])
        )
    for cls in [2]:
        gamma = tf.reduce_sum((1 - y_pred_channels[cls]) * y_true_channels[cls]) / (
            tf.reduce_sum(y_true_channels) + eps
        )
        # print('  False Negative Gamma:', gamma)
        loss = loss + gamma
    return tf.identity(loss, name="generalized_categorical_cross_entropy")


def weighted_categorical_crossentropy_with_fpr_loss(y_true, y_pred, distance=0.5, eps=1e-5):
    cat_cross_ent = tf.losses.categorical_crossentropy(y_true, y_pred)
    y_pred_bin = tf.argmax(y_pred, axis=-1)
    classes = y_pred.shape[-1]
    y_pred_channels = tf.split(y_pred, classes, axis=-1)
    y_true_channels = tf.split(y_true, classes, axis=-1)
    for cls in range(classes):
        c_true = tf.squeeze(y_true_channels[cls], -1)
        weight = 1.0 / (tf.reduce_sum(c_true) + eps)  # 1 / |Y_true|
        loss = (
            tf.reduce_sum(cat_cross_ent * c_true * weight)
            if cls == 0
            else loss + tf.reduce_sum(cat_cross_ent * c_true * weight)
        )

        # Calc. FP Rate Correction
        c_false_p = tf.cast(
            tf.squeeze(tf.not_equal(y_true_channels[cls], 1), -1), tf.float32
        ) * tf.cast(
            tf.equal(y_pred_bin, cls), tf.float32
        )  # Calculate false predictions
        gamma = 1 / classes + (
            tf.reduce_sum(
                tf.abs((c_false_p * tf.squeeze(y_pred_channels[cls], -1)) - distance)
            )
            / (tf.reduce_sum(c_false_p) + eps)
        )  # Calculate Gamma
        weight_c = weight * gamma  # gamma / |Y+|
        loss = loss + tf.reduce_sum(
            cat_cross_ent * c_false_p * weight_c
        )  # Add FP Correction

    return loss


def pearson_correlation_coefficient_loss(x, y):
    # https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras/46620771
    mean_x = tf.reduce_mean(x)
    mean_y = tf.reduce_mean(y)
    x_m, y_m = x - mean_x, y - mean_y
    r_num = tf.reduce_sum(tf.multiply(x_m, y_m))
    r_den = tf.sqrt(
        tf.multiply(tf.reduce_sum(tf.square(x_m)), tf.reduce_sum(tf.square(y_m)))
    )
    r = r_num / r_den
    r = tf.maximum(tf.minimum(r, 1.0), -1.0)
    return 1 - tf.square(r)
