"""Different metrics implemented in numpy and tensorflow and SimpleITK
For use in tensorflow, the metrics should be converted to metric objects.
"""
import math

import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from .loss import calculate_nmi, dice_coefficient

# pylint: disable=missing-function-docstring

### Metric classes


class Dice(Metric):  # adapted from tensorflow example implementation
    """This class implements the dice coefficient as a tensorflow metric"""

    def __init__(self, name="dice", num_classes=2, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dice = self.add_weight(name="dice", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

        self.num_classes = num_classes

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        """Do the actual calculations

        Parameters
        ----------
        y_true : tf.Tensor
            The ground truth
        y_pred : tf.Tensor
            The predicted tensor
        sample_weight : tf.Tensor, optional
            sample weights, not implemented at the moment, by default None
        """
        # call dice coefficient
        dice = dice_coefficient(y_true, y_pred)
        if sample_weight is not None:
            raise NotImplementedError("Different sample weights not implemented")
        # update dice
        self.dice.assign_add(dice)
        # update count
        self.count.assign_add(1)

    def result(self):
        return self.dice / self.count

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype, "num_classes": self.num_classes}


class NMI(Metric):
    """This class implements the normalized mutual information metric

    Global Mutual information loss for image-image pairs.

    Original Author: Courtney Guo
    If you use this loss function, please cite the following:
    Guo, Courtney K. Multi-modal image registration with unsupervised deep learning. MEng. Thesis
    Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces
    Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
    MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545

    Parameters
    ----------
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
        name=None,
        dtype=None,
        n_bins=150,
        min_val=-1,
        max_val=1,
        normalize=False,
        sigma_ratio=0.5,
        clip=True,
        include_endpoints=False,
        **kwargs,
    ):
        super().__init__(name, dtype, **kwargs)
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

        self.nmi = self.add_weight(name="nmi", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def get_config(self):
        return {
            "n_bins": self.n_bins,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "normalize": self.normalize,
            "sigma_ratio": self.sigma_ration,
            "clip": self.clip,
            "include_endpoints": self.include_endpoints,
        }

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        """Do the actual calculations

        Parameters
        ----------
        y_true : tf.Tensor
            The ground truth
        y_pred : tf.Tensor
            The predicted tensor
        sample_weight : tf.Tensor, optional
            sample weights, not implemented at the moment, by default None
        """
        # call dice coefficient
        nmi = calculate_nmi(
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
        if sample_weight is not None:
            raise NotImplementedError("Different sample weights not implemented")
        # update count
        self.count.assign_add(nmi.shape[0])
        # update nmi
        self.nmi.assign_add(tf.reduce_sum(nmi))

    def result(self):
        return self.nmi / self.count


### SITK
# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html


def get_ml_sitk(image):
    image_filter = sitk.StatisticsImageFilter()
    image_filter.Execute(image)
    spacing = image.GetSpacing()
    voxel = spacing[0] * spacing[1] * spacing[2]
    cubic_mm = image_filter.GetSum() * voxel
    volume = cubic_mm / 1000
    return volume


def get_connectivity_sitk(output):
    connected_component_filter = sitk.ConnectedComponentImageFilter()
    component_map = sitk.GetArrayFromImage(connected_component_filter.Execute(output))
    bin_count = np.bincount(component_map.flat)
    if len(bin_count) > 1:
        cc_id = np.argmax(bin_count[1:]) + 1
        return bin_count[cc_id] / np.sum(sitk.GetArrayFromImage(output))
    else:
        return 0


def get_fragmentation_sitk(output):
    connected_component_filter = sitk.ConnectedComponentImageFilter()
    component_map = sitk.GetArrayFromImage(connected_component_filter.Execute(output))
    nc_output = np.bincount(component_map.flat).size - 1
    if nc_output > 0:
        return 1 - 1 / nc_output
    else:
        return 1


def confusion_rate_sitk(output, target, target_class, non_target_class):
    prediction_non_target = output == non_target_class
    label_target = target == target_class
    confused = sitk.Multiply(prediction_non_target, label_target)
    stats_f = sitk.StatisticsImageFilter()
    stats_f.Execute(confused)
    n_confused = stats_f.GetSum()
    stats_f.Execute(label_target)
    n_target = stats_f.GetSum()
    if np.isclose(n_target, 0):
        return np.nan
    return n_confused / n_target


def hausdorff_metric_sitk(output, target):
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.Execute(target, output)
    return hausdorff_distance_filter.GetHausdorffDistance()


def overlap_measures_sitk(output, target):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(target, output)
    dice_coefficient_sitk = overlap_measures_filter.GetDiceCoefficient()
    volume_similarity = overlap_measures_filter.GetVolumeSimilarity()
    false_negative = overlap_measures_filter.GetFalseNegativeError()
    false_positive = overlap_measures_filter.GetFalsePositiveError()
    iou = overlap_measures_filter.GetJaccardCoefficient()
    return dice_coefficient_sitk, volume_similarity, false_negative, false_positive, iou


def symmetric_surface_measures_sitk(output, target):
    # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html
    if sitk.GetArrayFromImage(target).sum() == 0:
        return 0, 0, 0, 0
    segmented_surface = sitk.LabelContour(output)
    reference_surface = sitk.LabelContour(target)
    reference_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(target, squaredDistance=False)
    )
    segmented_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(output, squaredDistance=False)
    )
    seg2ref_distance_map = reference_distance_map * sitk.Cast(
        segmented_surface, sitk.sitkFloat32
    )
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(
        reference_surface, sitk.sitkFloat32
    )
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(
        np.zeros(num_segmented_surface_pixels - len(seg2ref_distances))
    )
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(
        np.zeros(num_reference_surface_pixels - len(ref2seg_distances))
    )

    all_surface_distances = seg2ref_distances + ref2seg_distances
    mean_symmetric_surface_distance = np.mean(all_surface_distances)
    median_symmetric_surface_distance = np.median(all_surface_distances)
    std_symmetric_surface_distance = np.std(all_surface_distances)
    max_symmetric_surface_distance = np.max(all_surface_distances)
    return (
        mean_symmetric_surface_distance,
        median_symmetric_surface_distance,
        std_symmetric_surface_distance,
        max_symmetric_surface_distance,
    )


### Numpy


def dice_coefficient_np(output, target, threshold=0.5, smooth=1e-5):
    output = output[:, :, :, 1] > threshold
    target = target[:, :, :, 1] > threshold
    union = np.count_nonzero(np.logical_and(output, target))
    label = np.count_nonzero(output)
    pred = np.count_nonzero(target)
    hard_dice = (2 * union + smooth) / (label + pred + smooth)
    return hard_dice


def signal_to_noise_ratio_np(output, object_mask, background_mask, eps=1e-5):
    dividend = np.mean(output[object_mask])
    divisor = np.std(output[background_mask])
    with np.errstate(divide="ignore"):
        return np.abs(dividend) / (divisor)


def contrast_to_noise_ratio_np(output, object_mask, background_mask, eps=1e-5):
    dividend = np.abs(np.mean(output[object_mask]) - np.mean(output[background_mask]))
    divisor = np.std(output[background_mask])
    with np.errstate(divide="ignore"):
        return np.abs(dividend) / (divisor)


def root_mean_squared_error_np(output, target):
    squared_error = (output.ravel() - target.ravel()) ** 2
    mean_squared_error = np.mean(squared_error)
    return np.sqrt(mean_squared_error)


def mean_squared_error_np(output, target):
    squared_error = (output.ravel() - target.ravel()) ** 2
    mean_squared_error = np.mean(squared_error)
    return mean_squared_error


def mean_absolute_error_np(output, target):
    absolute_error = np.abs(output.ravel() - target.ravel())
    mean_absolute_error = np.mean(absolute_error)
    return mean_absolute_error


def mutual_information_np(output, target, bins=200):
    # I(X, Y) = H(X) + H(Y) - H(X,Y)
    # https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
    output = output.ravel()
    target = target.ravel()

    correlation_xy = np.histogram2d(output, target, bins)[0]
    correlation_x = np.histogram(output, bins)[0]
    correlation_y = np.histogram(target, bins)[0]

    entropy_x = _shannon_entropy(correlation_x)
    entropy_y = _shannon_entropy(correlation_y)
    entropy_xy = _shannon_entropy(correlation_xy)

    mutual_information = entropy_x + entropy_y - entropy_xy
    return mutual_information


def normalized_mutual_information_np(output, target, bins=200):
    # symmetric uncertainty
    mutual_information = mutual_information_np(output, target, bins)
    correlation_x = np.histogram(output, bins)[0]
    correlation_y = np.histogram(target, bins)[0]
    entropy_x = _shannon_entropy(correlation_x)
    entropy_y = _shannon_entropy(correlation_y)
    return (2 * mutual_information) / (entropy_y + entropy_x)


def normalized_cross_correlation_2d_np(output, target, filter_dim=5):
    np.seterr(divide="ignore", invalid="ignore")
    s_f = np.ones((filter_dim, filter_dim))
    m_f = np.divide(s_f, (filter_dim**2))
    assert filter_dim % 2 != 0
    padding_size = math.ceil((filter_dim - 1) / 2)

    m_x = _convolution2d_np(
        np.pad(
            output, ((padding_size, padding_size), (padding_size, padding_size)), "edge"
        ),
        m_f,
    )
    m_x = output - m_x
    m_y = _convolution2d_np(
        np.pad(
            target, ((padding_size, padding_size), (padding_size, padding_size)), "edge"
        ),
        m_f,
    )
    m_y = target - m_y
    cc_map = np.divide(
        _convolution2d_np(np.multiply(m_x, m_y), s_f),
        np.sqrt(
            np.multiply(_convolution2d_np(m_x**2, s_f), _convolution2d_np(m_y**2, s_f))
        ),
    )
    cross_correlation = np.mean(cc_map)
    return cross_correlation


def cross_correlation_1d_np(output, target):
    output = output.ravel()
    target = target.ravel()
    cross_correlation = np.correlate(output, target)
    return cross_correlation


### Helper Functions


def _convolution2d_np(image, kernel, bias=1e-5):
    k_s = kernel.shape
    if k_s[0] == k_s[1]:
        i_s = image.shape
        y = i_s[0] - (k_s[0] - 1)
        x = i_s[1] - (k_s[0] - 1)
        new_image = np.zeros((y, x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = (
                    np.sum(image[i : i + k_s[0], j : j + k_s[0]] * kernel) + bias
                )
    return new_image


def _shannon_entropy(image):
    c_normalized = image / float(np.sum(image))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    entropy = -sum(c_normalized * np.log2(c_normalized))
    return entropy
