"""This model can be used to load images into tensorflow. The segbasisloader
will augment the images while the apply loader can be used to pass whole images.
"""
import logging
import os
from enum import Enum
from typing import List, Optional

import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from . import config as cfg
from . import normalization
from .NetworkBasis import image
from .NetworkBasis.dataloader import DataLoader
from .normalization import NORMALIZING

# configure logger
logger = logging.getLogger(__name__)

# define enums
class NOISETYP(Enum):
    """The different noise types"""

    GAUSSIAN = 0
    POISSON = 1


class SegBasisLoader(DataLoader):
    """A basis loader for segmentation network, should be extended for specific
    task implementing the _adapt_to_task method. The Image is padded by a quarter of
    the sample size in each direction and the random patches are extracted.

    If frac_obj is > 0, the specific fraction of the samples_per_volume will be
    selected, so that the center is on a foreground class. Due to the other samples
    being truly random, the fraction containing a sample can be higher, but this is
    not a problem if the foreground is strongly underrepresented. If this is not
    the case, the samples should be chosen truly randomly.

    Parameters
    ----------
    seed : int, optional
        set a fixed seed for the loader, by default 42
    mode : has no effect, should not be Apply
    name : str, optional
        the name of the loader, by default 'reader'
    frac_obj : float, optional
        The fraction of samples that should be taken from the foreground if None,
        the values set in the config file will be used, if set to 0, sampling
        will be completely random, by default None
    samples_per_volume : int, optional:
        The number of samples that should be taken from one volume per epoch.
    """

    def __init__(
        self,
        mode=None,
        seed=42,
        name="reader",
        frac_obj=None,
        samples_per_volume=None,
        normalizing_method=None,
        do_resampling=None,
        shuffle=None,
        sample_buffer_size=None,
        **kwargs,
    ):

        # set new properties derived in the shape
        self.data_rank = None

        super().__init__(
            mode=mode,
            seed=seed,
            name=name,
            shuffle=shuffle,
            sample_buffer_size=sample_buffer_size,
            **kwargs,
        )

        # use caching
        self.use_caching = True

        # get the fraction of the samples that should contain the object
        if frac_obj is None:
            self.frac_obj = cfg.percent_of_object_samples
        else:
            self.frac_obj = frac_obj

        # set the samples per volume
        if samples_per_volume is None:
            self.samples_per_volume = cfg.samples_per_volume
        else:
            self.samples_per_volume = samples_per_volume

        if normalizing_method is not None:
            self.normalizing_method = normalizing_method
        else:
            self.normalizing_method = cfg.normalizing_method

        if do_resampling is not None:
            self.do_resampling = do_resampling
        else:
            self.do_resampling = cfg.do_resampling

        # set channel and class parameters
        self.n_channels = cfg.num_channels
        self.n_seg = cfg.num_classes_seg

        # set the capacity
        if sample_buffer_size is None:
            self.sample_buffer_size = cfg.batch_capacity_train

        # set callbacks for normalization (in case it should be applied to the complete dataset)
        self.normalization_callbacks = []

        # TODO: turn normalization into an object
        # set the normalization_method
        if self.normalizing_method == NORMALIZING.QUANTILE:
            self.normalization_func = lambda img: normalization.normalie_channelwise(
                image=img,
                func=normalization.quantile,
                lower_q=cfg.norm_min_q,
                upper_q=cfg.norm_max_q,
            )
        elif self.normalizing_method == NORMALIZING.MEAN_STD:
            self.normalization_func = lambda img: normalization.normalie_channelwise(
                image=img, func=normalization.mean_std
            )
        elif self.normalizing_method == NORMALIZING.WINDOW:
            self.normalization_func = lambda img: normalization.normalie_channelwise(
                image=img,
                func=normalization.window,
                lower=cfg.norm_min_v,
                upper=cfg.norm_max_v,
            )
        elif self.normalizing_method == NORMALIZING.HISTOGRAM_MATCHING:
            # set file to export the landmarks to
            self.landmark_file = os.path.join(
                self._get_preprocessing_folder(), "landmarks.npz"
            )
            self.normalization_func = lambda img: normalization.histogram_matching_apply(
                self.landmark_file, img, None, True
            )
            if not os.path.exists(self.landmark_file):
                self.normalization_callbacks.append(
                    lambda x: normalization.landmark_and_mean_extraction(
                        self.landmark_file, x
                    )
                )
        elif self.normalizing_method == NORMALIZING.Z_SCORE:
            # set file to export the landmarks to (the same as for landmarks, because it extracts both)
            self.landmark_file = os.path.join(
                self._get_preprocessing_folder(), "landmarks.npz"
            )
            self.normalization_func = lambda img: normalization.z_score(
                self.landmark_file, img
            )
            if not os.path.exists(self.landmark_file):
                self.normalization_callbacks.append(
                    lambda x: normalization.landmark_and_mean_extraction(
                        self.landmark_file, x
                    )
                )
        elif self.normalizing_method == NORMALIZING.HM_QUANTILE:
            # normalize with quantile method before histogram matching
            n_func = lambda img: normalization.quantile(
                image=img, lower_q=cfg.norm_min_q, upper_q=cfg.norm_max_q
            )
            # set file to export the landmarks to
            self.landmark_file = os.path.join(
                self._get_preprocessing_folder(), "landmarks.npz"
            )
            self.normalization_func = lambda img: normalization.histogram_matching_apply(
                self.landmark_file, img, n_func, False
            )
            if not os.path.exists(self.landmark_file):
                self.normalization_callbacks.append(
                    lambda x: normalization.landmark_and_mean_extraction(
                        self.landmark_file, x, norm_func=n_func
                    )
                )
        elif self.normalizing_method == NORMALIZING.HM_QUANT_MEAN:
            # normalize with quantile method before histogram matching
            n_func = lambda img: normalization.quantile(
                image=img, lower_q=cfg.norm_min_q, upper_q=cfg.norm_max_q
            )
            # set file to export the landmarks to
            self.landmark_file = os.path.join(
                self._get_preprocessing_folder(), "landmarks.npz"
            )
            self.normalization_func = lambda img: normalization.histogram_matching_apply(
                self.landmark_file, img, n_func, True
            )
            if not os.path.exists(self.landmark_file):
                self.normalization_callbacks.append(
                    lambda x: normalization.landmark_and_mean_extraction(
                        self.landmark_file, x, norm_func=n_func
                    )
                )
        else:
            raise NotImplementedError(f"{self.normalizing_method} is not implemented")

    def __call__(
        self,
        file_list: List[str],
        batch_size: Optional[int] = None,
        n_epochs=50,
        read_threads=1,
        **kwargs,
    ) -> tf.data.Dataset:
        """Create the tensorflow dataset when calling the data loader

        Parameters
        ----------
        file_list : List[str]
            The files that should be loaded
        batch_size : int
            The batch size, if None, cfg.train_batch_size will be used, by default None
        n_epochs : int, optional
            The number of epochs. Each file will be used once per epoch, by default 50
        read_threads : int, optional
            The number of read threads, by default 1

        Returns
        -------
        tf.data.Dataset
            The tensorflow dataset, all files will be shuffled each epoch, then the
            samples will be interleaved and the dataset is shuffled again with a
            buffer 3 * number of files
        """
        # call the normalization callbacks, but only in training
        if self.mode == self.MODES.TRAIN:
            for n_call in self.normalization_callbacks:
                n_call([sitk.ReadImage(self.get_filenames(f)[0]) for f in file_list])
        if batch_size is None:
            batch_size = cfg.batch_size_train
        return super().__call__(
            file_list, batch_size, n_epochs=n_epochs, read_threads=read_threads
        )

    def _set_up_shapes_and_types(self):
        """
        sets all important configurations from the config file:
        - n_channels
        - dtypes
        - dshapes

        also derives:
        - data_rank
        - slice_shift

        """
        # dtypes and dshapes are defined in the base class
        # pylint: disable=attribute-defined-outside-init

        if self.mode is self.MODES.TRAIN or self.mode is self.MODES.VALIDATE:
            self.dtypes = [cfg.dtype, cfg.dtype]
            self.dshapes = [
                np.array(cfg.train_input_shape),
                np.array(cfg.train_label_shape),
            ]
            # use the same shape for image and labels
            assert np.all(
                self.dshapes[0][:2] == self.dshapes[1][:2]
            ), "Sample and label shapes do not match."
        else:
            raise ValueError(f"Not allowed mode {self.mode}")

        self.data_rank = len(self.dshapes[0])

        assert self.data_rank in [3, 4], "The rank should be 3 or 4."

    def get_filenames(self, file_id):
        """For compability reasons, get filenames without the preprocessed ones

        Parameters
        ----------
        file_id : str
            The file id

        Returns
        -------
        [type]
            [description]
        """
        sample, _, label, _ = self._get_filenames_cached(file_id)
        return sample, label

    def _get_preprocessing_folder(self):
        # set preprocessing folder name
        folder_name = f"pre_{self.normalizing_method.name}"
        if self.do_resampling:
            folder_name += "_resampled"
        pre_folder = os.path.join(cfg.preprocessed_dir, folder_name)
        if not os.path.exists(pre_folder):
            logger.debug("preprocessing dir does not exist, it will be created")
            os.makedirs(pre_folder)
        return pre_folder

    def _get_filenames_cached(self, file_id):
        """Gets the filenames and the preprocessed filenames

        Parameters
        ----------
        file_id : str
            The file ID

        Returns
        -------
        str, str, str, str
            The path to the data_file and label_file in the cached and uncached version
        """
        # get the folder and id
        folder, file_number = os.path.split(file_id)

        pre_folder = self._get_preprocessing_folder()

        # generate the file name for the sample
        sample_name = cfg.sample_file_name_prefix + file_number
        data_file = os.path.join(folder, sample_name + cfg.file_suffix)
        if not os.path.exists(data_file):
            raise Exception(f"The file {data_file} could not be found")
        # generate the name of the preprocessed file
        filename = f"{sample_name}.mhd"
        data_file_pre = os.path.join(pre_folder, filename)

        # generate the file name for the sample
        label_name = cfg.label_file_name_prefix + file_number
        label_file = os.path.join(folder, (label_name + cfg.file_suffix))
        if not os.path.exists(data_file):
            raise Exception(f"The file {label_file} could not be found")
        # generate the name of the preprocessed file
        label_file_pre = os.path.join(pre_folder, (label_name + ".mhd"))
        return data_file, data_file_pre, label_file, label_file_pre

    def _load_file(self, file_name, load_labels=True):
        """Load a file, if the file is found in the chache, this is used, otherwise
        the file is preprocessed and added to the chache

        Preprocessed files are saved as images, this increases the load time
        from 20 ms to 50 ms per image but is not really relevant compared to
        the sampleing time. The advantage is that SimpleITK can be used for
        augmentation, which does not work when storing numpy arrays.

        Parameters
        ----------
        file_name : str or bytes
            Filename must be either a string or utf-8 bytes as returned by tf.
        load_labels : bool, optional
            If true, the labels will also be loaded, by default True

        Returns
        -------
        data, lbl
            The preprocessed data and label files
        """

        # convert to string if necessary
        if isinstance(file_name, bytes):
            file_id = str(file_name, "utf-8")
        else:
            file_id = str(file_name)
        logger.debug("        Loading %s (%s)", file_id, self.mode)
        # Use a SimpleITK reader to load the nii images and labels for training
        data_file, data_file_pre, label_file, label_file_pre = self._get_filenames_cached(
            file_id
        )
        # see if the preprocessed files exist
        data_exist = os.path.exists(data_file_pre)
        lbl_exist = os.path.exists(label_file_pre)
        if data_exist and lbl_exist and self.use_caching and load_labels:
            # load images
            data = sitk.ReadImage(str(data_file_pre))
            lbl = sitk.ReadImage(str(label_file_pre))
        elif not load_labels and data_exist and self.use_caching:
            # load images
            data = sitk.ReadImage(str(data_file_pre))
            lbl = None
        else:
            # load images
            data_img = sitk.ReadImage(data_file)
            if load_labels:
                label_img = sitk.ReadImage(label_file)
            else:
                label_img = None
            assert not np.any(
                np.isnan(sitk.GetArrayFromImage(data_img))
            ), f"Nans after loading {data_file}"
            # adapt, resample and normalize them
            data_img, label_img = self.adapt_to_task(data_img, label_img)
            if self.do_resampling:
                if load_labels:
                    data_img, label_img = self._resample(data_img, label_img)
                else:
                    data_img = self._resample(data_img, label_img)
            data = sitk.GetArrayFromImage(data_img)
            assert not np.any(np.isnan(data)), "Nans in the image after converting to numpy"
            assert np.all(data < 1e6), "Input values over were 1 000 000 found."
            try:
                data = self.normalize(data)
                assert not np.any(np.isnan(data)), "Nans in the image after normalization"
            except AssertionError as exc:
                tf.print(f"There was the error {exc} when processing {data_file}.")
                raise exc
            if load_labels:
                lbl = sitk.GetArrayFromImage(label_img)
            else:
                lbl = None
            # then check them
            self._check_images(data, lbl)
            # if everything is ok, save the preprocessed images
            data_img_proc = sitk.GetImageFromArray(data)
            data_img_proc.CopyInformation(data_img)
            sitk.WriteImage(data_img_proc, str(data_file_pre))
            data = data_img_proc
            if load_labels:
                label_img_proc = sitk.GetImageFromArray(lbl)
                label_img_proc.CopyInformation(label_img)
                sitk.WriteImage(label_img_proc, str(label_file_pre))
                lbl = label_img_proc

        return data, lbl

    def adapt_to_task(self, data_img: sitk.Image, label_img: sitk.Image):
        """This function can be used to adapt the images to the task at hand.

        Parameters
        ----------
        data_img : sitk.Image
            The image
        label_img : sitk.Image
            The labels

        Returns
        -------
        sitk.Image, sitk.Image
            The adapted image and labels
        """
        return data_img, label_img

    def _check_images(self, data: np.ndarray, lbl: np.ndarray):
        """Check the images for problems

        Parameters
        ----------
        data : np.array
            Numpy array containing the data
        lbl : np.array
            Numpy array containing the labels (or None)
        """
        assert not np.any(np.isnan(data)), "Nans in the image in check"
        if lbl is not None:
            assert not np.any(np.isnan(lbl)), "Nans in the labels in check"
            assert np.sum(lbl) > 100, "Not enough labels in the image"
            logger.debug("Checking Labels (min, max) %s %s:", np.min(lbl), np.max(lbl))
            logger.debug("Shape Label: %s", lbl.shape)
        logger.debug("Shape Data: %s", data.shape)

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Normaize an image. The specified normalization method is used.

        Parameters
        ----------
        img : np.array, optional
            the image to normalize

        Returns
        -------
        np.array
            Normalized image
        """
        assert not np.any(np.isnan(img)), "NaNs in the input image."
        img_no_nan = np.nan_to_num(img, nan=cfg.data_background_value)

        # normalie the image
        image_normalized = self.normalization_func(img_no_nan)

        # do checks
        assert not np.any(np.isnan(image_normalized)), "NaNs in normalized image."
        assert np.abs(image_normalized).max() < 1e3, "Voxel values over 1000."
        return image_normalized

    def _resample(self, data: sitk.Image, label=None):
        """Resample the image and labels to the spacing specified in the config.
        The orientation is not changed, but it is checked, that all directions are
        perpendicular to each other.

        Parameters
        ----------
        data : sitk.Image
            The image
        label : sitk.Image, optional
            The labels, optional, by default None

        Returns
        -------
        sitk.Image
            One or two images, depending if labels are provided
        """
        data_info = image.get_data_info(data)
        # assure that the images are similar
        if label is not None:
            assert np.allclose(
                data_info["orig_direction"], label.GetDirection(), atol=0.01
            ), "label and image do not have the same direction"
            assert np.allclose(
                data_info["orig_origin"], label.GetOrigin(), atol=0.01
            ), "label and image do not have the same origin"
            assert np.allclose(
                data_info["orig_spacing"], label.GetSpacing(), atol=0.01
            ), "label and image do not have the same spacing"

        target_info = {}
        target_info["target_spacing"] = cfg.target_spacing

        # make sure the spacing is orthogonal
        orig_dirs = np.array(data_info["orig_direction"]).reshape(3, 3)
        assert np.isclose(
            np.dot(orig_dirs[0], orig_dirs[1]), 0, atol=0.01
        ), "x and y not orthogonal"
        assert np.isclose(
            np.dot(orig_dirs[0], orig_dirs[2]), 0, atol=0.01
        ), "x and z not orthogonal"
        assert np.isclose(
            np.dot(orig_dirs[1], orig_dirs[2]), 0, atol=0.01
        ), "y and z not orthogonal"
        target_info["target_direction"] = data_info["orig_direction"]

        # calculate the new size
        orig_size = np.array(data_info["orig_size"]) * np.array(data_info["orig_spacing"])
        target_info["target_size"] = list(
            (orig_size / np.array(cfg.target_spacing)).astype(int)
        )

        target_info["target_type_image"] = cfg.target_type_image
        target_info["target_type_label"] = cfg.target_type_label

        return image.resample_sitk_image(
            data,
            target_info,
            data_background_value=cfg.data_background_value,
            do_adapt_resolution=self.do_resampling,
            label=label,
            label_background_value=cfg.label_background_value,
            do_augment=False,
        )

    def _read_file_and_return_numpy_samples(self, file_name_queue: bytes):
        """Helper function getting the actual samples

        Parameters
        ----------
        file_name_queue : bytes
            The filename

        Returns
        -------
        np.array, np.array
            The samples and labels
        """
        data_img, label_img = self._load_file(file_name_queue)
        samples, labels = self._get_samples_from_volume(data_img, label_img)
        return samples, labels

    def _get_samples_from_volume(self, data_img: sitk.Image, label_img: sitk.Image):
        """This is where the sampling actually takes place. The images are first
        augmented using sitk functions and the augmented using numpy functions.
        Then they are converted to numpy array and sampled as described in the
        class description

        Parameters
        ----------
        data_img : sitk.Image
            The sample image
        label_img : sitk.Image
            The labels as integers

        Returns
        -------
        np.array, np.array
            The image samples and the lables as one hot labels

        Raises
        ------
        NotImplementedError
            If mode is apply, this is raised, the Apply loader should be used instead
        """

        if self.mode == self.MODES.APPLY:
            raise NotImplementedError("Use the original data loader")

        # augment whole images
        data_img, label_img = self._augment_images(data_img, label_img)

        # convert samples to numpy arrays
        data = sitk.GetArrayFromImage(data_img)
        lbl = sitk.GetArrayFromImage(label_img)

        # augment the numpy arrays
        data, lbl = self._augment_numpy(data, lbl)

        # check that there are labels
        assert np.any(lbl != 0), "no labels found"
        # check shape
        assert np.all(data.shape[:-1] == lbl.shape)
        assert len(data.shape) == 4, "data should be 4d"
        assert len(lbl.shape) == 3, "labels should be 3d"

        # determine the number of background and foreground samples
        n_foreground = int(self.samples_per_volume * self.frac_obj)
        n_background = int(self.samples_per_volume - n_foreground)

        # calculate the maximum padding, so that at least three quarters in
        # each dimension is inside the image
        # sample shape is without the number of channels
        if self.data_rank == 4:
            sample_shape = self.dshapes[0][:-1]
        # if the rank is three, add a dimension for the z-extent
        elif self.data_rank == 3:
            sample_shape = np.array(
                [
                    1,
                ]
                + list(self.dshapes[0][:2])
            )
        assert (
            sample_shape.size == len(data.shape) - 1
        ), "sample dims do not match data dims"
        max_padding = sample_shape // 4

        # pad the data (using 0s)
        pad_with = ((max_padding[0],) * 2, (max_padding[1],) * 2, (max_padding[2],) * 2)
        data_padded = np.pad(data, pad_with + ((0, 0),))
        label_padded = np.pad(lbl, pad_with)

        # calculate the allowed indices
        # the indices are applied to the padded data, so the minimum is 0
        # the last dimension, which is the number of channels is ignored
        min_index = np.zeros(3, dtype=int)
        # the maximum is the new data shape minus the sample shape (accounting for the padding)
        max_index = data_padded.shape[:-1] - sample_shape
        assert np.all(min_index <= max_index), "image to small to get patches"

        # create the arrays to store the samples
        batch_shape = (n_foreground + n_background,) + tuple(sample_shape)
        samples = np.zeros(batch_shape + (self.n_channels,), dtype=cfg.dtype_np)
        labels = np.zeros(batch_shape, dtype=np.uint8)

        # get the background origins (get thrice as many, in case they contain labels)
        # This is faster than drawing again each time
        background_shape = (3 * n_background, 3)
        origins_background = np.random.randint(
            low=min_index, high=max_index, size=background_shape
        )

        # get the foreground center
        valid_centers = np.argwhere(lbl)
        indices = np.random.randint(low=0, high=valid_centers.shape[0], size=n_foreground)
        origins_foreground = valid_centers[indices] + max_padding - sample_shape // 2
        # check that they are below the maximum amount of padding
        for i, m_index in enumerate(max_index):
            origins_foreground[:, i] = np.clip(origins_foreground[:, i], 0, m_index)

        # extract patches (pad if necessary), in separate function, do augmentation beforehand or with patches
        origins = np.concatenate([origins_foreground, origins_background])
        # count the samples
        num = 0
        for i, j, k in origins:
            sample_patch = data_padded[
                i : i + sample_shape[0], j : j + sample_shape[1], k : k + sample_shape[2]
            ]
            label_patch = label_padded[
                i : i + sample_shape[0], j : j + sample_shape[1], k : k + sample_shape[2]
            ]
            if num < n_foreground:
                samples[num] = sample_patch
                labels[num] = label_patch
                num += 1
            # only use patches with not too many labels
            elif label_patch.mean() < cfg.background_label_percentage:
                samples[num] = sample_patch
                labels[num] = label_patch
                num += 1
            # stop if there are enough samples
            if num >= self.samples_per_volume:
                break

        if num < self.samples_per_volume:
            raise ValueError(
                f"Could only find {num} samples, probably not enough background, consider not using ratio sampling "
                + "or increasing the background_label_percentage (especially for 3D)."
            )

        # if rank is 3, squash the z-axes
        if self.data_rank == 3:
            samples = samples.squeeze(axis=1)
            labels = labels.squeeze(axis=1)

        # convert to one_hot_label
        labels_onehot = np.squeeze(np.eye(self.n_seg)[labels.flat]).reshape(
            labels.shape + (-1,)
        )

        logger.debug(
            "Sample shape: %s, Label_shape: %s",
            str(samples.shape),
            str(labels_onehot.shape),
        )

        return samples, labels_onehot

    def _augment_numpy(self, img: np.ndarray, lbl: np.ndarray):
        """!
        samplewise data augmentation

        @param I <em>numpy array,  </em> image samples
        @param L <em>numpy array,  </em> label samples

        Three augmentations are available:
        - intensity variation
        """

        if cfg.add_noise and self.mode is self.MODES.TRAIN:
            if cfg.noise_typ == NOISETYP.GAUSSIAN:
                gaussian = np.random.normal(0, cfg.standard_deviation, img.shape)
                logger.debug("Minimum Gauss %.3f:", gaussian.min())
                logger.debug("Maximum Gauss %.3f:", gaussian.max())
                img = img + gaussian

            elif cfg.noise_typ == NOISETYP.POISSON:
                poisson = np.random.poisson(cfg.mean_poisson, img.shape)
                # scale according to the values
                poisson = poisson * -cfg.mean_poisson / (cfg.norm_max_v - cfg.norm_min_v)
                logger.debug("Minimum Poisson %.3f:", poisson.min())
                logger.debug("Maximum Poisson %.3f:", poisson.max())
                img = img + poisson

        return img, lbl

    def _augment_images(self, data: sitk.Image, label: sitk.Image):
        """Augment images using sitk. Right now, rotations and scale changes are
        implemented. The values are set in the config. Images should already be
        resampled.

        Parameters
        ----------
        data : sitk.Image
            the image
        label : sitk.Image
            the labels

        Returns
        -------
        sitk.Image, sitk.Image
            the augmented data and labels
        """
        data_info = image.get_data_info(data)

        target_info = {}
        target_info["target_spacing"] = data_info["orig_spacing"]
        target_info["target_direction"] = data_info["orig_direction"]

        # do not change the resolution
        target_info["target_size"] = data_info["orig_size"]
        target_info["target_type_image"] = cfg.target_type_image
        target_info["target_type_label"] = cfg.target_type_label

        if self.mode is self.MODES.TRAIN:
            return image.resample_sitk_image(
                data,
                target_info=target_info,
                data_background_value=cfg.data_background_value,
                do_adapt_resolution=False,
                label=label,
                label_background_value=cfg.label_background_value,
                do_augment=True,
                max_rotation_augment=cfg.max_rotation,
                min_resolution_augment=cfg.min_resolution_augment,
                max_resolution_augment=cfg.max_resolution_augment,
            )
        else:
            return data, label
