"""Different losses that can be used as functions in TensorFlow.
"""
from typing import Callable

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss

from . import metric as Metric

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


class MutualInformation(Loss):
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

    def entropy(self, prob: tf.Tensor):
        # entropy is - sum_i p_i log p_i
        product = -prob * K.log(prob + K.epsilon())
        return K.sum(K.sum(product, 1), 1)

    def call(self, y_true, y_pred):
        if self.clip:
            y_true = tf.clip_by_value(
                y_true, clip_value_min=self.min_val, clip_value_max=self.max_val
            )
            y_pred = tf.clip_by_value(
                y_pred, clip_value_min=self.min_val, clip_value_max=self.max_val
            )

        # reshape: flatten images into shape (batch_size, height x width x depth x chan, 1)
        y_true = tf.reshape(y_true, (-1, tf.math.reduce_prod(y_true.shape[1:])))
        y_true = tf.expand_dims(y_true, 2)
        y_pred = tf.reshape(y_pred, (-1, tf.math.reduce_prod(y_pred.shape[1:])))
        y_pred = tf.expand_dims(y_pred, 2)

        if self.debug:
            tf.debugging.assert_all_finite(y_true, message="NaNs in y_true")
            tf.debugging.assert_all_finite(y_pred, message="NaNs in y_pred")

        number_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)

        # reshape bin centers to be (1, 1, B)
        bin_shape = [1, 1, np.prod(self.bin_centers.get_shape().as_list())]
        bin_centers = tf.reshape(self.bin_centers, bin_shape)

        # calculate how much each voxel contributes to each intensity using
        # a gaussian weighting function
        matrix_a = tf.exp(-self.preterm * tf.square(y_true - bin_centers))
        # and normalize along bin dimension
        matrix_a_norm = K.sum(matrix_a, -1, keepdims=True)
        matrix_a /= matrix_a_norm

        matrix_b_dist = tf.exp(-self.preterm * tf.square(y_pred - bin_centers))
        matrix_b_norm = K.sum(matrix_b_dist, -1, keepdims=True)
        matrix_b = matrix_b_dist / matrix_b_norm

        if self.debug:
            tf.debugging.assert_all_finite(matrix_a, message="NaNs in matrix_a")
            tf.debugging.assert_all_finite(matrix_a_norm, message="NaNs in matrix_a_norm")
            tf.debugging.assert_all_finite(matrix_b_dist, message="NaNs in matrix_b_dist")
            tf.debugging.assert_all_finite(matrix_b, message="NaNs in matrix_b")
            tf.debugging.assert_all_finite(matrix_b_norm, message="NaNs in matrix_b_norm")

        # compute probabilities
        matrix_a_permuted = K.permute_dimensions(matrix_a, (0, 2, 1))
        prob_ab = K.batch_dot(
            matrix_a_permuted, matrix_b
        )  # should be the right size now, nb_labels x nb_bins
        prob_ab /= number_voxels
        prob_a = tf.reduce_mean(matrix_a, 1, keepdims=True)
        prob_b = tf.reduce_mean(matrix_b, 1, keepdims=True)

        if self.debug:
            tf.debugging.assert_all_finite(prob_a, message="NaNs in prob_a")
            tf.debugging.assert_all_finite(prob_b, message="NaNs in prob_b")
            tf.debugging.assert_all_finite(prob_ab, message="NaNs in prob_ab")

        prob_prod = (
            K.batch_dot(K.permute_dimensions(prob_a, (0, 2, 1)), prob_b) + K.epsilon()
        )
        nmi_loss = prob_ab * K.log(prob_ab / prob_prod + K.epsilon())

        if self.debug:
            tf.debugging.assert_all_finite(prob_prod, message="NaNs in prob_prod")
            tf.debugging.assert_all_finite(nmi_loss, message="NaNs in nmi_loss")

        mutual_info = K.sum(K.sum(nmi_loss, 1), 1)

        if self.normalize and mutual_info > K.epsilon():
            mutual_info = mutual_info / self.entropy(prob_a)
        return tf.clip_by_value(mutual_info, 0, np.inf, name=self.name)

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
