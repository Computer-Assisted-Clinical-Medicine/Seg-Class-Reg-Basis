"""This model can be used to load images into tensorflow. The segbasisloader
will augment the images while the apply loader can be used to pass whole images.
"""
import logging
import os
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from . import config as cfg
from .NetworkBasis.dataloader import DataLoader

# configure logger
logger = logging.getLogger(__name__)

# define enums
class NOISETYP(Enum):
    """The different noise types"""

    GAUSSIAN = 0
    POISSON = 1


class SegBasisLoader(DataLoader):
    """A basis loader for segmentation network. The Image is padded by a quarter of
    the sample size in each direction and the random patches are extracted.

    If frac_obj is > 0, the specific fraction of the samples_per_volume will be
    selected, so that the center is on a foreground class. Due to the other samples
    being truly random, the fraction containing a sample can be higher, but this is
    not a problem if the foreground is strongly underrepresented. If this is not
    the case, the samples should be chosen truly randomly.

    Parameters
    ----------
    file_dict : Dict[str, Dict[str, str]]
        dictionary containing the file information, the key should be the id of
        the data point and value should be a dict with the image and labels as
        keys and the file paths as values
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
        file_dict: Dict[str, Dict[str, str]],
        mode=None,
        seed=42,
        name="reader",
        frac_obj=None,
        samples_per_volume=None,
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

        # save the file dict
        self.file_dict = file_dict

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

        # set channel and class parameters
        self.n_channels = cfg.num_channels
        self.n_seg = cfg.num_classes_seg

        # set the capacity
        if sample_buffer_size is None:
            self.sample_buffer_size = cfg.batch_capacity_train

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
        str, str
            The location of the image and labels
        """
        data = self.file_dict[file_id]
        sample = os.path.join(cfg.data_base_dir, data["image"])
        assert os.path.exists(sample), "image not found."
        if "labels" in data:
            labels = os.path.join(cfg.data_base_dir, data["labels"])
            assert os.path.exists(labels), "labels not found."
        else:
            labels = None
        return sample, labels

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
        data_file, label_file = self.get_filenames(file_id)
        # load images
        data_img = sitk.ReadImage(str(data_file))
        if load_labels:
            label_img = sitk.ReadImage(str(label_file))
        else:
            label_img = None
        # adapt to task
        data_img, label_img = self.adapt_to_task(data_img, label_img)
        return data_img, label_img

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
        assert isinstance(data_img, sitk.Image), "data should be an SimpleITK image"
        assert isinstance(label_img, sitk.Image), "labels should be an SimpleITK image"
        # augment only in training
        if self.mode == self.MODES.TRAIN:
            data_img, label_img = self._augment_images(data_img, label_img)

        # convert samples to numpy arrays
        data = sitk.GetArrayFromImage(data_img)
        lbl = sitk.GetArrayFromImage(label_img)

        # augment the numpy arrays
        if self.mode == self.MODES.TRAIN:
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

        # if the image is too small otherwise, pad some more
        size_diff = sample_shape - (data.shape[:-1] + max_padding * 2)
        if np.any(size_diff >= 0):
            logger.debug(
                "Sample size to small with %s, padding will be increased", sample_shape
            )
            # add padding to the dimensions with a positive difference
            max_padding += np.ceil(np.maximum(size_diff, 0) / 2).astype(int)
            # add one extra to make it bigger
            max_padding += (size_diff >= 0).astype(int)

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
        assert np.all(min_index < max_index), (
            f"image to small too get patches size {data_padded.shape[:-1]} < sample "
            + f"shape {sample_shape} with padding {pad_with} and orig. size {data.shape[:-1]}"
        )

        # create the arrays to store the samples
        batch_shape = (n_foreground + n_background,) + tuple(sample_shape)
        samples = np.zeros(batch_shape + (self.n_channels,), dtype=cfg.dtype_np)
        labels = np.zeros(batch_shape, dtype=np.uint8)

        # get the background origins (get twice as many, in case they contain labels)
        # This is faster than drawing again each time
        background_shape = (2 * n_background, 3)
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
        origins = list(np.concatenate([origins_foreground, origins_background]))
        # count the samples
        num = 0
        counter = 0
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
            # add more samples if they threaten to run out
            counter += 1
            if counter == len(origins):
                origins += list(
                    np.random.randint(low=min_index, high=max_index, size=background_shape)
                )

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
                poisson = poisson * -cfg.mean_poisson
                logger.debug("Minimum Poisson %.3f:", poisson.min())
                logger.debug("Maximum Poisson %.3f:", poisson.max())
                img = img + poisson

        return img, lbl

    def _augment_images(self, image: sitk.Image, label: sitk.Image):
        """Augment images using sitk. Right now, rotations and scale changes are
        implemented. The values are set in the config. Images should already be
        resampled.

        Parameters
        ----------
        image : sitk.Image
            the image
        label : sitk.Image
            the labels

        Returns
        -------
        sitk.Image, sitk.Image
            the augmented data and labels
        """
        assert (
            self.mode is self.MODES.TRAIN
        ), "Augmentation should only be done in training mode"

        rotation = np.random.uniform(np.pi * -cfg.max_rotation, np.pi * cfg.max_rotation)

        transform = sitk.Euler3DTransform()
        # rotation center is center of the image center in world coordinates
        rotation_center = image.TransformContinuousIndexToPhysicalPoint(
            [i / 2 for i in image.GetSize()]
        )
        transform.SetCenter(rotation_center)
        logger.debug("Augment Rotation: %s", rotation)
        transform.SetRotation(0, 0, rotation)
        transform.SetTranslation((0, 0, 0))

        resolution_augmentation = np.random.uniform(
            low=cfg.min_resolution_augment, high=cfg.max_resolution_augment
        )
        aug_target_spacing = np.array(image.GetSpacing()) * resolution_augmentation

        # resample the image
        resample_method = sitk.ResampleImageFilter()
        resample_method.SetOutputSpacing(list(aug_target_spacing))
        resample_method.SetDefaultPixelValue(0)
        resample_method.SetInterpolator(sitk.sitkLinear)
        resample_method.SetOutputDirection(image.GetDirection())
        resample_method.SetOutputOrigin(image.GetOrigin())
        resample_method.SetSize(image.GetSize())
        resample_method.SetTransform(transform)
        # for some reason, Float 32 does not work
        resample_method.SetOutputPixelType(sitk.sitkFloat64)
        # it does not work for multiple components per pixel
        if image.GetNumberOfComponentsPerPixel() > 1:
            components = []
            for i in range(image.GetNumberOfComponentsPerPixel()):
                component = sitk.VectorIndexSelectionCast(image, i)
                assert (
                    component.GetNumberOfComponentsPerPixel() == 1
                ), "There only should be one component per pixel"
                components.append(resample_method.Execute(component))
            new_image = sitk.Compose(components)
        else:
            new_image = resample_method.Execute(image)

        if label is not None:
            # change setting for the label
            resample_method.SetInterpolator(sitk.sitkNearestNeighbor)
            resample_method.SetOutputPixelType(sitk.sitkUInt8)
            resample_method.SetDefaultPixelValue(0)
            # label: nearest neighbor resampling, fill with background
            new_label = resample_method.Execute(label)
        else:
            new_label = None

        return new_image, new_label
