"""Different losses that can be used as functions in TensorFlow.
"""
from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import losses

# pylint: disable=missing-function-docstring


def get_loss(loss_name: str, loss_parameters: dict = None) -> Callable:
    """
    Returns loss depending on loss.

    just look at the function to see the allowed losses

    Parameters
    ----------
    loss_name : str
        The name of the loss
    loss_parameters : dict, optional
        Additional parameters to be passed on, if None, no parameters will be
        passed, by default None.

    Returns
    -------
    Callable
        The loss as tensorflow function
    """
    # many returns do not affect the readability
    # pylint: disable=too-many-return-statements

    if loss_parameters is None:
        loss_parameters = {}

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

    if loss_name == "NMI":
        return MutualInformation(**loss_parameters)

    if loss_name == "CON-OUT":
        return ConstrainOutput(**loss_parameters)

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


def dice_coefficient(target, output, smooth=1e-5):
    """Calculate the dice coefficient, a small factor is applied for smoothing to prevent
    numerical issues."""
    axis = tf.range(0, tf.rank(output) - 1)
    output = tf.cast(output, dtype=tf.float32)
    target = tf.cast(target, dtype=tf.float32)
    intersection = tf.reduce_sum(output * target, axis=axis)
    union = tf.reduce_sum(output, axis=axis) + tf.reduce_sum(target, axis=axis)
    hard_dice = (2.0 * intersection + smooth) / (union + smooth)
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice


def soft_dice_coefficient(target, output, smooth=1e-5):
    axis = tf.range(0, tf.rank(output) - 1)
    output = tf.cast(output, dtype=tf.float32)
    target = tf.cast(target, dtype=tf.float32)
    intersection = tf.reduce_sum(output * target, axis=axis)
    union = tf.reduce_sum(tf.square(output), axis=axis) + tf.reduce_sum(
        tf.square(target), axis=axis
    )
    soft_dice = (2.0 * intersection + smooth) / (union + smooth)
    soft_dice = tf.reduce_mean(soft_dice)
    return soft_dice


def tanimoto_coefficient(output, target, smooth=1e-5):
    axis = tf.range(0, tf.rank(output) - 1)
    output = tf.cast(output, dtype=tf.float32)
    target = tf.cast(target, dtype=tf.float32)
    intersection = tf.reduce_sum(output * target, axis=axis)
    union = tf.reduce_sum(output, axis=axis) + tf.reduce_sum(target, axis=axis)
    hard_dice = (intersection + smooth) / (union + smooth)
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice


def dice_loss(y_true, y_pred, eps=1e-12):
    loss = tf.subtract(
        tf.constant(1, tf.float32), dice_coefficient(y_true, y_pred), name="dice"
    )  # 1-dice to have a loss that can be minimized
    return loss


def soft_dice_loss(y_true, y_pred, smooth=1e-6):
    loss = tf.subtract(
        tf.constant(1, tf.float32),
        soft_dice_coefficient(y_true, y_pred, smooth=smooth),
        name="dice",
    )  # 1-dice to have a loss that can be minimized
    return loss


def dice_with_fnr_loss(y_true, y_pred, eps=1e-12):
    loss = tf.subtract(
        tf.constant(1, tf.float32), dice_coefficient(y_true, y_pred), name="dice"
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


def entropy(prob: tf.Tensor):
    # entropy is - sum_i p_i log p_i
    product = -prob * K.log(prob + K.epsilon())
    return K.sum(K.sum(product, 1), 1)


def calculate_nmi(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    bin_centers: tf.Tensor,
    preterm: tf.Tensor,
    normalize=True,
    name=None,
    clip=False,
    min_val=-1,
    max_val=1,
) -> tf.Tensor:
    """Calculate the (normalized) mutual information

    Original Author: Courtney Guo
    If you use this loss function, please cite the following:
    Guo, Courtney K. Multi-modal image registration with unsupervised deep learning. MEng. Thesis
    Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces
    Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
    MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545

    Parameters
    ----------
    y_true : tf.Tensor
        The true image
    y_pred : tf.Tensor
        The predicted
    bin_centers : tf.Tensor
        The centers of the bins used for nmi calculation
    preterm : tf.Tensor
        The preterm for the distance calculation (sigma * constants)
    normalize : bool, optional
        If the MI should be normalized, by default True
    name : str, optional
        The name, if none it will be nmi or mi depending on the norm, by default None
    clip : bool, optional
        If the input should be clipped (too far away from bin center creates Nans), by default False
    min_val : int, optional
        The minimum clip value (should be lowest end of the bins), by default -1
    max_val : int, optional
        The maximum clip value (should be the highest end of the bins), by default 1

    Returns
    -------
    tf.tensor
        The calculated (N)MI loss
    """
    if name is None:
        if normalize:
            name = "nmi"
        else:
            name = "mi"
    if clip:
        y_true = tf.clip_by_value(y_true, clip_value_min=min_val, clip_value_max=max_val)
        y_pred = tf.clip_by_value(y_pred, clip_value_min=min_val, clip_value_max=max_val)

    if not y_true.dtype == y_pred.dtype:
        y_true = tf.cast(y_true, y_pred.dtype)
    if not bin_centers.dtype == y_pred.dtype:
        bin_centers = tf.cast(bin_centers, y_pred.dtype)
    if not preterm.dtype == y_pred.dtype:
        preterm = tf.cast(preterm, y_pred.dtype)

    # reshape: flatten images into shape (batch_size, height x width x depth x chan, 1)
    y_true = tf.reshape(y_true, (-1, tf.math.reduce_prod(y_true.shape[1:])))
    y_true = tf.expand_dims(y_true, 2)
    y_pred = tf.reshape(y_pred, (-1, tf.math.reduce_prod(y_pred.shape[1:])))
    y_pred = tf.expand_dims(y_pred, 2)

    number_voxels = tf.cast(K.shape(y_pred)[1], y_pred.dtype)

    # reshape bin centers to be (1, 1, B)
    bin_shape = [1, 1, np.prod(bin_centers.get_shape().as_list())]
    bin_centers = tf.reshape(bin_centers, bin_shape)

    # calculate how much each voxel contributes to each intensity using
    # a gaussian weighting function
    matrix_a = tf.exp(-preterm * tf.square(y_true - bin_centers))
    # and normalize along bin dimension
    matrix_a_norm = K.sum(matrix_a, -1, keepdims=True)
    matrix_a /= matrix_a_norm

    matrix_b_dist = tf.exp(-preterm * tf.square(y_pred - bin_centers))
    matrix_b_norm = K.sum(matrix_b_dist, -1, keepdims=True)
    matrix_b = matrix_b_dist / matrix_b_norm

    # compute probabilities
    matrix_a_permuted = K.permute_dimensions(matrix_a, (0, 2, 1))
    prob_ab = K.batch_dot(
        matrix_a_permuted, matrix_b
    )  # should be the right size now, nb_labels x nb_bins
    prob_ab /= number_voxels
    prob_a = tf.reduce_mean(matrix_a, 1, keepdims=True)
    prob_b = tf.reduce_mean(matrix_b, 1, keepdims=True)

    prob_prod = K.batch_dot(K.permute_dimensions(prob_a, (0, 2, 1)), prob_b) + K.epsilon()
    nmi_loss = prob_ab * K.log(prob_ab / prob_prod + K.epsilon())

    mutual_info = K.sum(K.sum(nmi_loss, 1), 1)

    if normalize and mutual_info > K.epsilon():
        mutual_info = mutual_info / entropy(prob_a)
    result = tf.clip_by_value(mutual_info, 0, np.inf, name=name)
    return tf.cast(result, tf.float32)


class MutualInformation(losses.Loss):
    """
    Global Mutual information loss for image-image pairs.

    Original Author: Courtney Guo
    If you use this loss function, please cite the following:
    Guo, Courtney K. Multi-modal image registration with unsupervised deep learning. MEng. Thesis
    Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces
    Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
    MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545

    Parameters
    ----------
    reduction : auto, optional
        Set by tensorflow, options are 'auto', 'none', 'sum', 'sum_over_batch_size', by default auto
    name : str, optional
        The name of the loss, by default "MI"
    n_bins : int, optional
        The number of bins, by default 150
    min_val : int, optional
        The minimum value of the data range (used to define the bins), by default -1
    max_val : int, optional
        The maximum data value, by default 1
    normalize : bool, optional
        If the loss should be normalized, it will be divided by the entropy of the
        ground truth image, by default False
    sigma_ratio : float, optional
        Scales how much the contribution is to other bins, by default 0.5
    clip : bool, optional
        Clip the images to the range defined by min_val and max_val, otherwise, there
        can be numerical errors, if a value is not in any of the bins
    include_endpoints : bool, optional
        If the endpoints should be included in the bin centers. This should be used
        for discrete values. By default False.
    """

    def __init__(
        self,
        reduction="auto",
        name="MI",
        n_bins=150,
        min_val=-1,
        max_val=1,
        normalize=False,
        sigma_ratio=0.5,
        clip=True,
        include_endpoints=False,
        debug=False,
    ):
        super().__init__(reduction, name)
        self.n_bins = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.normalize = normalize
        if sigma_ratio < 1e-6:
            raise ValueError("Sigma ration cannot be too close to 0")
        self.sigma_ration = sigma_ratio
        self.clip = clip
        self.include_endpoints = include_endpoints
        if self.include_endpoints:
            bin_centers = np.linspace(min_val, max_val, n_bins, endpoint=True)
        else:
            bin_centers = np.linspace(min_val, max_val, n_bins, endpoint=False)
            bin_centers = bin_centers + (bin_centers[1] - bin_centers[0]) / 2
        self.bin_centers = tf.constant(bin_centers, dtype=tf.float32)
        self.sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        self.preterm = tf.constant(1 / (2 * np.square(self.sigma)), dtype=tf.float32)
        self.debug = debug

    def call(self, y_true, y_pred):
        # make in negative, so that lower values are better
        return -calculate_nmi(
            y_true=y_true,
            y_pred=y_pred,
            bin_centers=self.bin_centers,
            preterm=self.preterm,
            normalize=self.normalize,
            name=self.name,
            clip=self.clip,
            min_val=self.min_val,
            max_val=self.max_val,
        )

    def get_config(self):
        return {
            "n_bins": self.n_bins,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "normalize": self.normalize,
            "sigma_ratio": self.sigma_ration,
            "clip": self.clip,
            "include_endpoints": self.include_endpoints,
            "reduction": self.reduction,
            "name": self.name,
        }


class ConstrainOutput(losses.Loss):
    """Constrain the output to be between two values, so the loss will be zero,
    if it is between the two values, otherwise, it will be scaling * overlap.

    Parameters
    ----------
    reduction : auto, optional
        Set by tensorflow, options are 'auto', 'none', 'sum', 'sum_over_batch_size', by default auto
    name : str, optional
        The name of the loss, by default "ConstrainOutput"
    min_val : float, optional
        The minimum value, by default 0.0
    max_val : float, optional
        The maximum value, by default 1.0
    scaling : float, optional
        By which value to scale the loss, by default 1
    """

    def __init__(
        self,
        reduction="auto",
        name="ConstrainOutput",
        min_val=0.0,
        max_val=1.0,
        scaling=1.0,
    ):
        self.min_val = min_val
        self.min_val_tensor = tf.convert_to_tensor(self.min_val, dtype=tf.float32)
        self.max_val = max_val
        self.max_val_tensor = tf.convert_to_tensor(self.max_val, dtype=tf.float32)
        self.scaling = scaling
        self.scaling_tensor = tf.convert_to_tensor(self.scaling, dtype=tf.float32)
        if self.min_val >= self.max_val:
            raise ValueError("Minimum value should be smaller than the maximum value")
        super().__init__(reduction, name)

    def call(self, _, y_pred):
        lower_constraint = tf.nn.relu(self.min_val_tensor - y_pred)
        upper_constraint = tf.nn.relu(y_pred - self.max_val_tensor)
        return self.scaling_tensor * (lower_constraint + upper_constraint)

    def get_config(self):
        return {
            "reduction": self.reduction,
            "name": self.name,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "scaling": self.scaling,
        }
