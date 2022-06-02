"""
Different methods to normalize the input images
"""
import logging
import os
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import yaml
from scipy.interpolate import interp1d
from tqdm import tqdm

# configure logger
logger = logging.getLogger(__name__)


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


class Normalization:
    """This is the base class for image normalization. The normalization_func has
    to be defined, which will do the normalization. train_normalization can also
    be implemented, which can be used to determine some parameters used for the
    normalization. properties_to_save should contain a list of all properties that
    should be loaded and saved to get the normalization method back and parameters_to_save
    should contain all parameters needed to create the normalization.
    """

    enum: Enum

    properties_to_save: List[str] = []
    parameters_to_save: List[str] = []

    def __init__(self, normalize_channelwise=True) -> None:
        self.normalize_channelwise = normalize_channelwise

    def normalization_func(self, image: np.ndarray) -> np.ndarray:
        return image

    def train_normalization(self, images: Iterable[sitk.Image]) -> None:
        pass

    def normalize(self, image: sitk.Image) -> sitk.Image:
        """Normalize an image. If normalize_channelwise is True, the normalization
        will be done channelwise, otherwise, it will be applied to the whole image.

        Parameters
        ----------
        image : sitk.Image
            The image to normalize

        Returns
        -------
        np.array
            The normalized image
        """
        image_np = sitk.GetArrayFromImage(image)
        if image_np.ndim == 4 and self.normalize_channelwise:
            if image_np.shape[3] > 1:
                for i in range(image_np.shape[3]):
                    # normalize each channel separately
                    image_np[:, :, :, i] = self.normalization_func(image_np[:, :, :, i])
            else:
                # otherwise, normalize the whole image
                image_np = self.normalization_func(image_np)
        else:
            # otherwise, normalize the whole image
            image_np = self.normalization_func(image_np)

        self.check_image(image_np)

        # turn back into image
        image_normalized = sitk.GetImageFromArray(image_np)
        image_normalized.CopyInformation(image)
        return image_normalized

    def check_image(self, image: np.ndarray) -> None:
        # do checks
        assert not np.any(np.isnan(image)), "NaNs in normalized image."
        assert np.abs(image).max() < 1e3, "Voxel values over 1000."

    def get_parameters(self) -> Dict[str, Any]:
        """get the parameters used to initialize the method, they are converted
        to lists if they are numpy arrays (for nicer exports)"""
        parameters: Dict[str, Any] = {}
        for param in self.parameters_to_save:
            val = getattr(self, param)
            if isinstance(val, np.ndarray):
                val = val.tolist()
            parameters[param] = val
        return parameters

    def __call__(self, image: sitk.Image) -> sitk.Image:
        return self.normalize(image)

    def to_file(self, file_path: os.PathLike):
        """Write all important properties and parameters to a file

        Parameters
        ----------
        file_path : os.PathLike
            The file to save to
        """
        properties: Dict[str, Any] = {}
        for prop in self.properties_to_save:
            # can't properly dump numpy arrays, so turn them into lists.
            val = getattr(self, prop)
            if isinstance(val, np.ndarray):
                val = val.tolist()
            properties[prop] = val
        data = {
            "parameters": self.get_parameters(),
            "properties": properties,
            "class": str(self.__class__),
            "enum": self.enum,
        }
        with open(file_path, "w", encoding="utf8") as f:
            yaml.dump(data, f)

    @classmethod
    def from_file(cls, file_path: os.PathLike):
        """Load the properties of the normalization from a file, it should
        already be initialized.

        Parameters
        ----------
        file_path : os.pathlike
            The file to load
        """
        with open(file_path, "r", encoding="utf8") as f:
            data = yaml.load(f, Loader=yaml.Loader)
        normalization = cls(**data["parameters"])
        # set all properties from the dict
        for prop, val in data["properties"].items():
            setattr(normalization, prop, val)
        return normalization


class NORMALIZING(Enum):
    """The different normalization types
    To get the corresponding class, call get_class
    """

    WINDOW = 0
    MEAN_STD = 1
    QUANTILE = 2
    HISTOGRAM_MATCHING = 3
    Z_SCORE = 4
    HM_QUANTILE = 5

    def get_class(self) -> Normalization:
        """Get the corresponding normalization class for an enum, it has to be a subclass
        of the Normalization class.

        Parameters
        ----------
        enum : NORMALIZING
            The enum

        Returns
        -------
        Normalization
            The normalization class

        Raises
        ------
        ValueError
            If the class was found for that enum
        """
        for norm_cls in all_subclasses(Normalization):
            if norm_cls.enum is self:
                return norm_cls
        raise ValueError(f"No normalization for {self.value}")


# define a few functions used by multiple normalizations
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

    return image_normed  # type: ignore


class Window(Normalization):
    """Normalize the image by a window. The image is clipped to the lower and
    upper value and then scaled to a range between -1 and 1.

    Parameters
    ----------
    lower : float
        The lower window value
    upper : float
        The higher value
    normalize_channelwise : bool
        If it should be applied channelwise, does not matter for window.
    """

    enum = NORMALIZING.WINDOW

    parameters_to_save = ["lower", "upper", "normalize_channelwise"]

    def __init__(self, lower: float, upper: float, normalize_channelwise: bool) -> None:
        self.lower = lower
        self.upper = upper
        if lower >= upper:
            raise ValueError("Lower has to be lower than upper")
        super().__init__(normalize_channelwise=normalize_channelwise)

    def normalization_func(self, image: np.ndarray) -> np.ndarray:
        """Normalize the image by a window. The image is clipped to the lower and
        upper value and then scaled to a range between -1 and 1.

        Parameters
        ----------
        image : np.array
            The image as numpy array

        Returns
        -------
        np.array
            The normalized image
        """
        return window(image=image, lower=self.lower, upper=self.upper)


class Quantile(Normalization):
    """Normalize the image by quantiles. The image is clipped to the lower and
    upper quantiles of the image and then scaled to a range between -1 and 1.

    Parameters
    ----------
    lower_q : float
        The lower quantile (between 0 and 1)
    upper_q : float
        The upper quantile (between 0 and 1)
    normalize_channelwise : bool
        If it should be applied channelwise
    """

    enum = NORMALIZING.QUANTILE

    parameters_to_save = ["lower_q", "upper_q", "normalize_channelwise"]

    def __init__(self, lower_q: float, upper_q: float, normalize_channelwise=True) -> None:
        self.lower_q = lower_q
        self.upper_q = upper_q
        super().__init__(normalize_channelwise=normalize_channelwise)

    def normalization_func(self, image: np.ndarray) -> np.ndarray:
        """Normalize the image by quantiles. The image is clipped to the lower and
        upper quantiles of the image and then scaled to a range between -1 and 1.

        Parameters
        ----------
        image : np.array
            The image as numpy array

        Returns
        -------
        np.array
            The normalized image
        """
        assert (
            self.upper_q > self.lower_q
        ), "Upper quantile has to be larger than the lower."
        assert (
            np.sum(np.isnan(image)) == 0
        ), f"There are {np.sum(np.isnan(image)):.2f} NaNs in the image."

        a_min = np.quantile(image, self.lower_q)
        a_max = np.quantile(image, self.upper_q)

        assert a_max > a_min, "Both quantiles are the same."

        return window(image, a_min, a_max)


class MeanSTD(Normalization):
    """Subtract the mean from image and the divide it by the standard deviation
    generating a value similar to the Z-Score.

    Parameters
    ----------
    mask_quantile : float
        Quantile to use for the lower bound of the mask when calculating the std
    normalize_channelwise : bool
        If it should be applied channelwise
    """

    enum = NORMALIZING.MEAN_STD

    parameters_to_save = ["mask_quantile"]

    def __init__(self, mask_quantile=0.05, normalize_channelwise=True) -> None:
        self.mask_quantile = mask_quantile
        super().__init__(normalize_channelwise=normalize_channelwise)

    def normalization_func(self, image: np.ndarray) -> np.ndarray:
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
        # clip image
        image = clip_outliers(image, 0.01, 0.95)
        # set mask if not present
        mask_data = image > np.quantile(image, self.mask_quantile)
        # apply mask
        image_masked = image[mask_data]
        image_normed = image - np.mean(image)
        std = np.std(image_masked)
        assert std > 0, "The standard deviation of the image is 0."
        image_normed = image_normed / std

        return image_normed


class HistogramMatching(Normalization):
    """Apply histogram matching to an image. The provided quantiles will be used
    as landmarks to match the histograms. The values will also be clipped to be
    within the lowest and highest landmark. This only works for 3D images.

    Parameters
    ----------
    quantiles : List[float]
        corresponding landmark quantiles to extract, if None, 1%, 99% and steps of
        10% between 10% and 90% will be used.
    mask_quantile : float
        Values below this quantile will be ignored, by default 0
    center_volume : List[float]
        A volume with this size in the center will be used to extract the landmarks,
        if None, [150, 150, 80] is used, by default None
    """

    enum = NORMALIZING.HISTOGRAM_MATCHING

    parameters_to_save = ["quantiles", "mask_quantile", "center_volume"]
    properties_to_save = ["standard_scale", "means", "stds"]

    def __init__(
        self, quantiles: List[float] = None, mask_quantile=0, center_volume=None
    ) -> None:
        if quantiles is None:
            quantiles = np.concatenate(([0.01], np.arange(0.10, 0.91, 0.10), [0.99]))
        self.quantiles = np.array(quantiles)
        if center_volume is None:
            center_volume = [180, 180, 100]
        self.center_volume = np.array(center_volume)
        if not np.all(self.quantiles >= 0):
            raise ValueError("All quantiles should be >= 0.")
        if not np.all(self.quantiles <= 1):
            raise ValueError("All quantiles should be <= 1.")
        if not np.all(self.quantiles[1:] - self.quantiles[:-1] > 0):
            raise ValueError("Quantiles should be ascending.")
        self.mask_quantile = mask_quantile

        # set properties that will be determined during normalization training
        self.standard_scale: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None

        super().__init__(normalize_channelwise=True)

    def get_landmarks(self, image: sitk.Image) -> Tuple[np.ndarray, float, float]:
        """
        get the landmarks for the Nyul and Udupa norm method for a specific image

        Parameters
        ----------
        img : sitk.Image
            image on which to find landmarks

        Returns
        -------
        np.ndarray
            intensity values corresponding to percentiles in img
        np.ndarray
            mean of the image
        np.ndarray
            std of the image
        """

        # crop image to main region
        n_voxels = np.round(self.center_volume / image.GetSpacing()).astype(int)
        # make sure the region is smaller than the image
        n_voxels = np.minimum(n_voxels, image.GetSize())
        start = (image.GetSize() - n_voxels) // 2
        end = start + n_voxels
        assert np.all(start >= 0), "start should be 0 or higher."
        assert np.all(end <= image.GetSize()), "The end should not be larger than the size."
        image = image[[slice(s, e) for s, e in zip(start, end)]]
        # get data
        img_data = sitk.GetArrayFromImage(image)
        assert img_data.ndim == 3, "Only 3D images are supported"
        # extract the channel and clip the outliers
        image_mod = clip_outliers(img_data, self.quantiles[0], self.quantiles[-1])
        # set mask if not present
        mask_data = image_mod > np.quantile(image_mod, self.mask_quantile)
        # apply mask
        masked = image_mod[mask_data]

        # get landmarks
        landmarks = np.quantile(masked, self.quantiles)
        # get mean
        mean = image_mod.mean()
        # get std
        std = image_mod.std()

        return landmarks, mean, std

    def train_normalization(self, images: Iterable[sitk.Image]) -> None:
        """Extract the mean and landmarks (quantiles) and the standard deviation
        from a set of images.
        """
        # initialize the scale
        standard_scale_list = []
        means_list = []
        stds_list = []
        for image in tqdm(images, unit="image", desc="Train Normalization"):
            # get landmarks
            landmarks, current_mean, current_std = self.get_landmarks(image)

            # gather landmarks for standard scale
            standard_scale_list.append(landmarks)
            means_list.append(float(current_mean))
            stds_list.append(float(current_std))

        mean_standard_scale = np.mean(np.array(standard_scale_list), axis=0)
        assert isinstance(mean_standard_scale, np.ndarray), "mean scale should be an array"
        self.standard_scale = mean_standard_scale
        self.means = np.array(means_list)
        self.stds = np.array(stds_list)

    def normalize(self, image: sitk.Image) -> sitk.Image:
        """Apply the histogram matching to an image

        Parameters
        ----------
        image : sitk.Image
            The image

        Returns
        -------
        sitk.Image
            The normalized image
        """

        # get landmarks
        landmarks, _, _ = self.get_landmarks(image)

        image_np = sitk.GetArrayFromImage(image)
        # get the clipped image
        image_clipped = clip_outliers(image_np, self.quantiles[0], self.quantiles[-1])

        # create interpolation function (with extremes of standard scale as fill values)
        f = interp1d(
            landmarks,
            self.standard_scale,
            fill_value=(self.standard_scale[0], self.standard_scale[-1]),
            bounds_error=False,
        )

        # apply it
        image_np = f(image_clipped)

        # rescale to a range between -1 and 1
        image_np = (
            2
            * (image_np - self.standard_scale[0])
            / (self.standard_scale[-1] - self.standard_scale[0])
            - 1
        )

        assert np.abs(image_np).max() < 10, "Voxel values over 10 detected"
        image_normalized = sitk.GetImageFromArray(image_np)
        image_normalized.CopyInformation(image)
        return image_normalized


class ZScore(HistogramMatching):
    """Apply the z_score normalization to an image. This means the mean of all
    images will be subtracted and then the values will be divided by the
    standard deviation. This is always done channelwise.

    Parameters
    ----------
    min_q : float
        The minimum quantile to use, values below will be clipped, by default 0
    max_q : float
        The maximum quantile to use, values above will be clipped, by default 1
    """

    enum = NORMALIZING.Z_SCORE

    parameters_to_save = ["min_q", "max_q"]
    properties_to_save = ["means", "stds"]

    def __init__(self, min_q: float = 0, max_q: float = 1) -> None:
        self.min_q = min_q
        self.max_q = max_q
        super().__init__(quantiles=[self.min_q, 0.5, self.max_q], mask_quantile=0)

    def normalize(self, image: sitk.Image) -> sitk.Image:
        """Apply the Z-Score normalization to an image

        Parameters
        ----------
        image : sitk.Image
            The image

        Returns
        -------
        sitk.Image
            The normalized image
        """

        image_np = sitk.GetArrayFromImage(image)

        assert self.means is not None, "No training was performed."
        assert self.stds is not None, "No training was performed."

        for i in range(image_np.shape[3]):
            mean = self.means[i]
            std = self.stds[i]

            # get the clipped image
            image_clipped = clip_outliers(image[:, :, :, i], self.min_q, self.max_q)

            # apply it
            image_np[:, :, :, i] = (image_clipped - mean) / std

        image_normalized = sitk.GetImageFromArray(image_np)
        image_normalized.CopyInformation(image)
        return image_normalized


class HMQuantile(HistogramMatching):
    """Apply histogram matching to an image. The provided quantiles will be used
    as landmarks to match the histograms. Previous to the histogram matching,
    quantile normalization will be performed. The lowest and highest quantiles
    in the quantiles will be used.

    Parameters
    ----------
    quantiles : List[float]
        corresponding landmark quantiles to extract, if None, 1%, 99% and steps of
        10% between 10% and 90% will be used.
    mask_quantile : float
        Values below this quantile will be ignored, by default 0
    """

    enum = NORMALIZING.HM_QUANTILE

    def __init__(
        self, quantiles: List[float] = None, mask_quantile=0, center_volume=None
    ) -> None:
        if quantiles is None:
            quantiles = np.concatenate(([0.01], np.arange(0.10, 0.91, 0.10), [0.99]))
        self.quantiles = np.array(quantiles)
        super().__init__(
            quantiles=quantiles, mask_quantile=mask_quantile, center_volume=center_volume
        )
        self.quantile_norm = Quantile(
            lower_q=self.quantiles[0],
            upper_q=self.quantiles[-1],
            normalize_channelwise=True,
        )

    def train_normalization(self, images: Iterable[sitk.Image]) -> None:
        # apply quantile normalization
        images = (self.quantile_norm(img) for img in images)
        return super().train_normalization(images)

    def normalize(self, image: sitk.Image) -> sitk.Image:
        image = self.quantile_norm(image)
        return super().normalize(image)
