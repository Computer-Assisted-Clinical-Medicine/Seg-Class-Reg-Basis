"""
Different methods to normalize the input images
"""
from enum import Enum
import os
from typing import Callable, List

import numpy as np
import SimpleITK as sitk
from scipy.interpolate import interp1d


class NORMALIZING(Enum):
    """The different normalization types"""

    WINDOW = 0
    MEAN_STD = 1
    QUANTILE = 2
    HISTOGRAM_MATCHING = 3
    Z_SCORE = 4
    HM_QUANTILE = 5
    HM_QUANT_MEAN = 6


def normalie_channelwise(image: np.ndarray, func: Callable, *args, **kwargs):
    """Normalize an image channelwise. All arguments besides image and func are
    passed onto the normalization function.

    Parameters
    ----------
    image : np.array
        The image to normalize
    func : Callable
        The function used for normalization.

    Returns
    -------
    np.array
        The normalized image
    """
    image_normed = image.copy()
    # see if there are multiple channels
    if image.ndim == 4:
        if image.shape[3] > 1:
            for i in range(image.shape[3]):
                # normalize each channel separately
                image_normed[:, :, :, i] = func(image[:, :, :, i], *args, **kwargs)
            return image_normed
        else:
            # otherwise, normalie the whole image
            return func(image, *args, **kwargs)
    else:
        # otherwise, normalie the whole image
        return func(image, *args, **kwargs)


def clip_outliers(image: np.ndarray, lower_q: float, upper_q: float):
    """Clip the outliers above and below a certain quantile.

    Parameters
    ----------
    image : np.array
        The image to clip.
    lower_q : float
        The lower quantile
    upper_q : float
        The upper quantile

    Returns
    -------
    np.array
        The image with the outliers removed.
    """

    # get the quantiles
    a_min = np.quantile(image, lower_q)
    a_max = np.quantile(image, upper_q)
    # clip
    return np.clip(image, a_min=a_min, a_max=a_max)


def window(image: np.ndarray, lower: float, upper: float):
    """Normalize the image by a window. The image is clipped to the lower and
    upper value and then scaled to a range between -1 and 1.

    Parameters
    ----------
    image : np.array
        The image as numpy array
    lower : float
        The lower value to clip at
    upper : float
        The higher value to clip at

    Returns
    -------
    np.array
        The normalized image
    """
    # clip
    image_normed = np.clip(image, a_min=lower, a_max=upper)
    # rescale to between 0 and 1
    image_normed = (image_normed - lower) / (upper - lower)
    # rescale to between -1 and 1
    image_normed = (image_normed * 2) - 1

    return image_normed


def quantile(image: np.ndarray, lower_q: float, upper_q: float) -> np.ndarray:
    """Normalize the image by quantiles. The image is clipped to the lower and
    upper quantiles of the image and then scaled to a range between -1 and 1.

    Parameters
    ----------
    image : np.array
        The image as numpy array
    lower_q : float
        The lower quantile (between 0 and 1)
    upper_q : float
        The upper quantile (between 0 and 1)

    Returns
    -------
    np.array
        The normalized image
    """
    assert upper_q > lower_q, "Upper quantile has to be larger than the lower."
    assert (
        np.sum(np.isnan(image)) == 0
    ), f"There are {np.sum(np.isnan(image)):.2f} NaNs in the image."

    a_min = np.quantile(image, lower_q)
    a_max = np.quantile(image, upper_q)

    assert a_max > a_min, "Both quantiles are the same."

    return window(image, a_min, a_max)


def mean_std(image: np.ndarray) -> np.ndarray:
    """Subtract the mean from image and the divide it by the standard deviation
    generating a value similar to the Z-Score.

    Parameters
    ----------
    image : np.array
        The image to normalize

    Returns
    -------
    np.array
        The normalized image
    """
    image_normed = image - np.mean(image)
    std = np.std(image_normed)
    assert std > 0, "The standard deviation of the image is 0."
    image_normed = image_normed / std

    return image_normed


def get_landmarks(img: np.ndarray, percs: np.ndarray) -> np.ndarray:
    """
    get the landmarks for the Nyul and Udupa norm method for a specific image

    Parameters
    ----------
    img : np.ndarray
        image on which to find landmarks
    percs : np.ndarray
        corresponding landmark quantiles to extract

    Returns
    -------
    np.ndarray
        intensity values corresponding to percs in img
    """
    landmarks = np.quantile(img, percs)
    return landmarks


def landmark_and_mean_extraction(
    landmark_file,
    images: List[sitk.Image],
    mask_percentile=0,
    percs=np.concatenate(([0.01], np.arange(0.10, 0.91, 0.10), [0.99])),
    norm_func=None,
):
    """Extract the mean and landmarks (quantiles) and the standard deviation
    from a set of images and export it to a files.

    Parameters
    ----------
    landmark_file : str
        The file where the features are saved
    images : List[sitk.Image]
        The list of images where the landmarks get extracted from
    mask_percentile : int, optional
        The percentile to use as mask value, by default 10
    percs : np.array, optional
        the quantiles to use for the landmarks, by default
        np.concatenate(([.05], np.arange(.10, .91, .10), [.95]))
    norm_func : function, optional
        An optional function which will be applied to each image before feature
        extraction, by default None
    """
    # initialize the scale
    standard_scale_list = []
    means_list = []
    stds_list = []
    for image in images:
        # get data
        img_data = sitk.GetArrayFromImage(image)
        # iterate over channels
        landmarks = []
        current_mean = []
        current_std = []
        for i in range(img_data.shape[3]):
            # extract the channel and clip the outliers
            image_mod = clip_outliers(img_data[:, :, :, i], percs[0], percs[-1])
            # normalize if specified
            if norm_func is not None:
                image_mod = norm_func(image_mod)
            # set mask if not present
            mask_data = image_mod > np.percentile(image_mod, mask_percentile)
            # apply mask
            masked = image_mod[mask_data]
            # get landmarks
            landmarks.append(get_landmarks(masked, percs))
            # get mean
            current_mean.append(image_mod.mean())
            # get std
            current_std.append(image_mod.std())

        # gather landmarks for standard scale
        standard_scale_list.append(landmarks)
        means_list.append(current_mean)
        stds_list.append(current_std)
    standard_scale = np.array(standard_scale_list)
    means = np.array(means_list)
    stds = np.array(stds_list)
    np.savez(
        landmark_file,
        percs=percs,
        standard_scale=standard_scale.mean(axis=0),
        all_landmarks=standard_scale,
        means=means,
        stds=stds,
        means_mean=means.mean(axis=0),
        stds_mean=stds.mean(axis=0),
        mask_percentile=mask_percentile,
    )


def histogram_matching_apply(
    landmark_file, image: np.ndarray, norm_func=None, subtract_mean=False
) -> np.ndarray:
    """Apply the histogram matching to an image

    Parameters
    ----------
    landmark_file : str
        The file where the landmarks are saved
    image : np.array
        The image
    norm_func : function, optional
        An optional function which will be applied to each image before histogram
        matching, by default None
    subtract_mean : bool, optional
        If this is true, the mean will be subtracted and all images will be
        divided by the standard deviation, by default false

    Returns
    -------
    np.array
        The normalized image

    Raises
    ------
    FileNotFoundError
        If the landmark file was not found
    """
    if not os.path.exists(landmark_file):
        raise FileNotFoundError(f"The landmark file {landmark_file} was not found.")
    # load landmark file
    data_file = np.load(landmark_file)
    percs = data_file["percs"]
    mask_percentile = data_file["mask_percentile"]

    image_normed = np.copy(image)

    for i in range(image.shape[3]):
        standard_scale = data_file["standard_scale"][i]

        # redefine standard scale
        if subtract_mean:
            standard_range = standard_scale[-1] - standard_scale[0]
            standard_scale = (
                2
                * (standard_scale - standard_scale[standard_scale.size // 2])
                / standard_range
            )

        # get the clipped image
        image_clipped = clip_outliers(image[:, :, :, i], percs[0], percs[-1])
        # normalize if specified
        if norm_func is not None:
            image_clipped = norm_func(image_clipped)
        # get mask
        mask_image = image_clipped > np.percentile(image_clipped, mask_percentile)
        # apply mask
        masked = image_clipped[mask_image]

        # get landmarks
        landmarks = get_landmarks(masked, percs)

        # create interpolation function (with extremes of standard scale as fill values)
        f = interp1d(
            landmarks,
            standard_scale,
            fill_value=(standard_scale[0], standard_scale[-1]),
            bounds_error=False,
        )

        # apply it
        image_normed[:, :, :, i] = f(image_clipped)

    assert np.abs(image_normed).max() < 1e3, "Voxel values over 1000 detected"
    return image_normed


def z_score(landmark_file, image: np.ndarray) -> np.ndarray:
    """Apply the z_score normalization to an image. This means the mean of all
    images will be subtracted and then the values will be divided by the
    standard deviation.

    Parameters
    ----------
    landmark_file : str
        The file where the landmarks are saved
    image : np.array
        The image

    Returns
    -------
    np.array
        The normalized image

    Raises
    ------
    FileNotFoundError
        If the landmark file was not found
    """
    if not os.path.exists(landmark_file):
        raise FileNotFoundError(f"The landmark file {landmark_file} was not found.")
    # load landmark file
    data_file = np.load(landmark_file)
    percs = data_file["percs"]

    image_normed = np.copy(image)

    for i in range(image.shape[3]):
        mean = data_file["means_mean"][i]
        std = data_file["stds_mean"][i]

        # get the clipped image
        image_clipped = clip_outliers(image[:, :, :, i], percs[0], percs[-1])

        # apply it
        image_normed[:, :, :, i] = (image_clipped - mean) / std

    return image_normed
