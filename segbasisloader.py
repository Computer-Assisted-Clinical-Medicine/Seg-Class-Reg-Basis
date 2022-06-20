"""This model can be used to load images into tensorflow. The segbasisloader
will augment the images while the apply loader can be used to pass whole images.
"""
import logging
import os
from collections.abc import Collection
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from . import config as cfg

# configure logger
logger = logging.getLogger(__name__)

# define enums
class NOISETYPE(Enum):
    """The different noise types"""

    GAUSSIAN = 0
    POISSON = 1


class SegBasisLoader:
    """A basis loader for segmentation network. The Image is padded by a quarter of
    the sample size in each direction and the random patches are extracted.

    If frac_obj is > 0, the specific fraction of the samples_per_volume will be
    selected, so that the center is on a foreground class. Due to the other samples
    being truly random, the fraction containing a sample can be higher, but this is
    not a problem if the foreground is strongly underrepresented. If this is not
    the case, the samples should be chosen truly randomly.

    Parameters
    ----------
    file_dict : Dict[str, Dict[str, Any]]
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
    shuffle : bool, optional:
        If the dataset should be shuffled. If None, it will be set to true when
        training and false for application, by default None
    sample_buffer_size : int, optional
        How big the buffer should be for shuffling, by default 4000.
    tasks : Tuple[str], optional
        Which tasks to perform, available are segmentation, classification, regression,
        autoencoder, by default ("segmentation",).
    """

    class MODES(Enum):
        """!
        Possible Modes for Reader Classes:
        - TRAIN = 'train': from list of file names, data augmentation and shuffling, drop remainder
        - VALIDATE = 'validate': from list of file names, shuffling, drop remainder
        - APPLY = 'apply' from list single file id, in correct order, remainder in last smaller batch
        """

        TRAIN = "train"
        VALIDATE = "validate"
        APPLY = "apply"

    def __init__(
        self,
        file_dict: Dict[str, Dict[str, Any]],
        mode=None,
        seed=42,
        name="reader",
        frac_obj=0.5,
        samples_per_volume=64,
        shuffle=None,
        sample_buffer_size=4000,
        tasks=("segmentation",),
        **kwargs,
    ):

        # set new properties derived in the shape
        self.data_rank = None

        # save the file dict
        self.file_dict = file_dict

        if mode is None:
            self.mode = self.MODES.TRAIN
        else:
            self.mode = mode

        if self.mode not in self.MODES:
            raise ValueError(f"mode = '{mode}' is not supported by network")

        self.seed = seed
        self.name = name

        self.frac_obj = frac_obj
        self.samples_per_volume = samples_per_volume

        if shuffle is None:
            if mode == self.MODES.APPLY:
                self.shuffle = False
            else:
                self.shuffle = True
        else:
            self.shuffle = shuffle

        if mode == self.MODES.APPLY and self.shuffle:
            raise ValueError("For applying, shuffle should be turned off.")

        self.sample_buffer_size = sample_buffer_size

        self.tasks = tasks

        # set channel and class parameters
        self.n_channels = cfg.num_channels
        self.n_seg = cfg.num_classes_seg

        # set the number of label images, classification and regression
        self.n_label_images = 0
        self.n_classification = 0
        self.n_regression = 0

        self.dshapes = None
        self.dtypes = None
        self.n_files = None

        self.n_inputs = None

        # set seed
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        self._set_up_shapes_and_types()

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

        if self.mode not in (self.MODES.TRAIN, self.MODES.VALIDATE):
            raise ValueError(f"Not allowed mode {self.mode}")

        # use the same shape for image and labels
        image_shape = tuple(cfg.train_input_shape)

        self.dshapes = []
        self.dtypes = []

        first = list(self.file_dict.values())[0]
        if not "image" in first:
            raise ValueError("This loader is made for images and expects one.")
        if isinstance(first["image"], (list, tuple)):
            n_images = len(first["image"])
        else:
            n_images = 1
        self.dshapes += [image_shape] * n_images
        self.dtypes += [cfg.dtype] * n_images
        self.n_inputs = n_images

        self.n_labels = 0
        if "segmentation" in self.tasks:
            labels_shape = tuple(cfg.train_label_shape)
            assert np.all(
                np.array(image_shape[:2]) == labels_shape[:2]
            ), "Sample and label shapes do not match."
            if isinstance(first["labels"], (list, tuple)):
                self.dshapes += [labels_shape] * len(first["labels"])
                self.dtypes += [cfg.dtype] * len(first["labels"])
                self.n_labels += len(first["labels"])
                self.n_label_images = len(first["labels"])
            else:
                self.dshapes.append(labels_shape)
                self.dtypes.append(cfg.dtype)
                self.n_labels += 1
                self.n_label_images = 1
        if "classification" in self.tasks:
            self.dshapes += [i.shape for i in first["classification"]]
            self.dtypes += [cfg.dtype] * len(first["classification"])
            self.n_labels += len(first["classification"])
            self.n_classification = len(first["classification"])
        if "regression" in self.tasks:
            self.dshapes += [(1,)] * len(first["regression"])
            self.dtypes += [cfg.dtype] * len(first["regression"])
            self.n_labels += len(first["regression"])
            self.n_regression = len(first["regression"])
        if "autoencoder" in self.tasks:
            # use the image shape and type again
            self.dshapes += [image_shape] * n_images
            self.dtypes += [cfg.dtype] * n_images
            self.n_labels += n_images

        assert len(self.dshapes) == len(self.dtypes)
        assert np.all([isinstance(i, tuple) for i in self.dshapes])

        self.data_rank = len(image_shape)
        assert self.data_rank in [3, 4], "The rank should be 3 or 4."

    def __call__(self, file_list: List, batch_size: int, n_epochs=50) -> tf.data.Dataset:
        """This function operates as follows,
        - Generates Tensor of strings from the file_list
        - Creates file_list_ds dataset from the Tensor of strings.
        - If loader is in training mode (self.mode == 'train'),
            - file_list_ds is repeated with shuffle n_epoch times
            - file_list_ds is mapped with _read_wrapper() to obtain the dataset.
            The mapping generates a set of samples from each pair of data and label files identified by each file ID.
            Here each element of dataset is a set of samples corresponding to one ID.
            - dataset is flat mapped to make each element correspond to one sample inside the dataset.
            - dataset is shuffled
            - dataset is batched with batch_size
            - 1 element of dataset is prefetched
            - dataset is returned

        - Else if loader is in validation mode (self.mode == 'validation'),
            - file_list_ds is mapped with _read_wrapper() to obtain dataset (mapping is same as train mode)
            - dataset is flat mapped to make each element correspond to one sample inside the dataset.
            - dataset is batched with batch_size
            - dataset is returned

        Parameters
        ----------
        file_list : List
            array of strings, where each string is a file ID corresponding to a pair of
            data file and label file to be loaded. file_list should be obtained from a .csv file
            and then converted to numpy array. Each ID string should have the format 'Location\\file_number'.
            From Location, the data file and label file with the file_number, respectively
            named as volume-file_number.nii and segmentation-file_number.nii are loaded.
            (See also LitsLoader._read_file(), LitsLoader._load_file() for more details.)
        batch_size : int
            The batch size
        n_epochs : int, optional
            The number of training epochs, by default 50

        Returns
        -------
        tf.data.Dataset
            tf.dataset of data and labels
        """

        if not np.issubdtype(type(batch_size), int):
            raise ValueError("The batch size should be an integer")

        if self.mode is self.MODES.APPLY:
            self.n_files = 1
        else:
            self.n_files = len(file_list)

        # set the buffer size
        if self.sample_buffer_size is None:
            sample_buffer_size = 8 * self.n_files
        else:
            sample_buffer_size = self.sample_buffer_size

        with tf.name_scope(self.name):
            id_tensor = tf.convert_to_tensor(file_list, dtype=tf.string)

            # Create dataset from list of file names
            if self.mode is self.MODES.APPLY:
                file_list_ds = tf.data.Dataset.from_tensors(id_tensor)
            else:
                file_list_ds = tf.data.Dataset.from_tensor_slices(id_tensor)

            if self.mode is self.MODES.TRAIN:
                # shuffle and repeat n_epoch times if in training mode
                file_list_ds = file_list_ds.shuffle(buffer_size=self.n_files).repeat(
                    count=n_epochs
                )

            # read data from file using the _read_wrapper
            dataset = file_list_ds.map(
                map_func=self._read_wrapper,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            # in the result, each element in the dataset has the number of samples
            # per file as first dimension followed by the sample shape

            # this function will flatten the datasets, so that each element has
            # the shape of the sample
            dataset = self._zip_data_elements_tensorwise(dataset)

            # zip the datasets, so that it is in the format (x, y)
            if self.n_inputs != 1 or self.n_labels > 1:
                dataset = dataset.map(self._make_x_y)

            if self.mode is not self.MODES.APPLY:
                # shuffle
                if self.shuffle:
                    dataset = dataset.shuffle(
                        buffer_size=sample_buffer_size, seed=self.seed
                    )

            if self.mode is self.MODES.APPLY:
                dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
            else:
                # no smaller final batch
                dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

            # batch prefetch
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def _zip_data_elements_tensorwise(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """here each element corresponds to one file.
        Use flat map to make each element correspond to one Sample.
        If there is more than one element, they will be zipped together. This could
        be (sample, label) or more elements.

        Parameters
        ----------
        ds : tf.data.Dataset
            Dataset

        Returns
        -------
        ds : tf.data.Dataset
            Dataset where each element corresponds to one sample

        """
        if len(dataset.element_spec) == 1:
            dataset = dataset.flat_map(lambda e: tf.data.Dataset.from_tensor_slices(e))
        else:
            # interleave the datasets, so that the order is the first sample from
            # all images, then the second one and so on. This results in better
            # shuffling, because not all sample from one image follow each other.
            dataset = dataset.interleave(
                lambda *elem: tf.data.Dataset.zip(
                    tuple((tf.data.Dataset.from_tensor_slices(e) for e in elem))
                )
            )
        return dataset

    def _make_x_y(self, *datasets) -> List[Any]:
        """This function takes a variable number of arguments and turns them into
        a list with 2 elements for x and y depending on the specified number of
        inputs and labels

        Returns
        -------
        List[Any]
            The list
        """
        output = []
        # if there is more than 1 input, turn it into a tuple
        if self.n_inputs > 1:
            output.append(tuple((ds for ds in datasets[: self.n_inputs])))
        else:
            output.append(datasets[0])

        # if there is more than 1 label, turn it into a tuple
        if self.n_labels == 1:
            output.append(datasets[self.n_inputs])
        elif self.n_labels > 1:
            output.append(
                tuple(
                    (ds for ds in datasets[self.n_inputs : self.n_inputs + self.n_labels])
                )
            )
        return output

    @tf.autograph.experimental.do_not_convert
    def _read_wrapper(self, id_data_set: List[tf.Tensor]) -> List[tf.Tensor]:
        """Wrapper for the _read_file() function
        Wraps the _read_file() function and handles tensor shapes and data types
        this has been adapted from https://github.com/DLTK/DLTK

        Parameters
        ----------
        id_data_set : list
            list of tf.Tensors from the id_list queue. Provides an identifier for the examples to read.
        kwargs :
            additional arguments for the '_read_sample function'

        Returns
        -------
        list
            list of tf.Tensors read for this example
        """

        def get_sample_tensors_from_file_name(file_id):
            """Wrapper for the python function
            Handles the data types of the py_func

            Parameters
            ----------
            file_id : list
            list of tf.Tensors from the id_list queue. Provides an identifier for the examples to read.

            Returns
            -------
            list
                list of things just read


            """
            try:
                samples_np = self._read_file_and_return_numpy_samples(file_id.numpy())
            except Exception as exc:
                logger.exception("got error %s from _read_file: %s", exc, file_id)
                print(f"Error when reading {file_id}")
                raise
            return samples_np

        ex = tf.py_function(
            get_sample_tensors_from_file_name, [id_data_set], self.dtypes
        )  # use python function in tensorflow

        tensors = []
        # set shape of tensors for downstream inference of shapes
        for tensor_shape, sample_shape in zip(ex, self.dshapes):
            if isinstance(sample_shape, Collection):
                shape: List[Any] = [None] + list(sample_shape)
            else:
                assert sample_shape == 1, "If shape is not 1, use an iterable."
                shape = [None, 1]
            tensor_shape.set_shape(shape)
            tensors.append(tensor_shape)

        return tensors

    def get_filenames(self, file_id):
        """For compatibility reasons, get filenames without the preprocessed ones

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

    def _load_file(self, file_name, load_labels=True, **kwargs):
        """Load a file

        Additional keyword arguments are passed to self.get_filenames

        Preprocessed files are saved as images, this increases the load time
        from 20 ms to 50 ms per image but is not really relevant compared to
        the sampling time. The advantage is that SimpleITK can be used for
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
        data_file, label_file = self.get_filenames(file_id, **kwargs)
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
        load_labels = self.n_label_images > 0
        data_img, label_img = self._load_file(
            file_name=file_name_queue, load_labels=load_labels
        )
        sample_np = self._get_samples_from_volume(data_img, label_img)
        sample_class_reg = self._get_class_reg(file_name_queue)
        # pass the sample to the autoencoder
        sample_autoencoder = self._get_autoencoder(file_name_queue, sample_np[0])
        return sample_np + sample_class_reg + sample_autoencoder

    def _get_samples_from_volume(
        self, data_img: sitk.Image, label_img: Optional[sitk.Image] = None
    ):
        """This is where the sampling actually takes place. The images are first
        augmented using sitk functions and the augmented using numpy functions.
        Then they are converted to numpy array and sampled as described in the
        class description

        Parameters
        ----------
        data_img : sitk.Image
            The sample image
        label_img : sitk.Image, optional
            The labels as integers

        Returns
        -------
        np.ndarray, np.ndarray
            The image samples and the labels as one hot labels, if no label_image
            is provided, the output is (np.ndarray,)
        """
        # TODO: change this function, so that an arbitrary number of images can be used

        # augment whole images
        assert isinstance(data_img, sitk.Image), "data should be an SimpleITK image"
        assert (
            isinstance(label_img, sitk.Image) or label_img is None
        ), "labels should be an SimpleITK image or None"
        # augment only in training
        if self.mode == self.MODES.TRAIN:
            data_img, label_img = self._augment_images(data_img, label_img)

        if label_img is None and self.frac_obj > 0:
            raise ValueError("Ratio Sampling only works if a label image is provided")

        # convert samples to numpy arrays
        data = sitk.GetArrayFromImage(data_img)
        # add 4th dimension if it is not there
        if data.ndim == 3:
            data = np.expand_dims(data, axis=-1)
        if label_img is None:
            lbl = None
        else:
            lbl = sitk.GetArrayFromImage(label_img)
            assert np.any(lbl != 0), "no labels found after sITK augmentation"

        # augment the numpy arrays
        if self.mode == self.MODES.TRAIN:
            data, lbl = self._augment_numpy(data, lbl)

        # check that there are labels
        assert np.any(lbl != 0), "no labels found after numpy augmentation"
        # check shape
        assert len(data.shape) == 4, "data should be 4d"
        if lbl is not None:
            assert len(lbl.shape) == 3, "labels should be 3d"
            assert np.all(data.shape[:-1] == lbl.shape), f"{data.shape} - {lbl.shape}"

        # determine the number of background and foreground samples
        n_foreground = int(self.samples_per_volume * self.frac_obj)
        n_background = int(self.samples_per_volume - n_foreground)

        # calculate the maximum padding, so that at least three quarters in
        # each dimension is inside the image
        # sample shape is without the number of channels
        if self.data_rank == 4:
            sample_shape = np.array(self.dshapes[0][:-1])
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
        if lbl is not None:
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
        if lbl is not None:
            labels = np.zeros(batch_shape, dtype=np.uint8)

        # get the background origins (get twice as many, in case they contain labels)
        # This is faster than drawing again each time
        background_shape = (2 * n_background, 3)
        origins_background = np.random.randint(
            low=min_index, high=max_index, size=background_shape
        )

        # get the foreground center
        valid_centers = np.argwhere(lbl)
        if n_foreground > 0:
            indices = np.random.randint(
                low=0, high=valid_centers.shape[0], size=n_foreground
            )
            origins_foreground = valid_centers[indices] + max_padding - sample_shape // 2
            # check that they are below the maximum amount of padding
            for i, m_index in enumerate(max_index):
                origins_foreground[:, i] = np.clip(origins_foreground[:, i], 0, m_index)
        else:
            origins_foreground = np.array([], dtype=int).reshape((0, 3))

        # extract patches (pad if necessary), in separate function, do augmentation beforehand or with patches
        origins = list(np.concatenate([origins_foreground, origins_background]))
        # count the samples
        num = 0
        counter = 0
        for i, j, k in origins:
            sample_patch = data_padded[
                i : i + sample_shape[0], j : j + sample_shape[1], k : k + sample_shape[2]
            ]
            if lbl is not None:
                label_patch = label_padded[
                    i : i + sample_shape[0],
                    j : j + sample_shape[1],
                    k : k + sample_shape[2],
                ]
            if num < n_foreground:
                samples[num] = sample_patch
                labels[num] = label_patch
                num += 1
            elif lbl is None:
                samples[num] = sample_patch
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
            if lbl is not None:
                labels = labels.squeeze(axis=1)

        # convert to one_hot_label
        if lbl is not None:
            labels_onehot = np.squeeze(np.eye(self.n_seg)[labels.flat]).reshape(
                labels.shape + (-1,)
            )

        if lbl is not None:
            logger.debug(
                "Sample shape: %s, Label_shape: %s",
                str(samples.shape),
                str(labels_onehot.shape),
            )
            return samples, labels_onehot
        else:
            logger.debug(
                "Sample shape: %s",
                str(samples.shape),
            )
            return (samples,)

    def _augment_numpy(self, img: np.ndarray, lbl: np.ndarray):
        """!
        samplewise data augmentation

        @param I <em>numpy array,  </em> image samples
        @param L <em>numpy array,  </em> label samples

        Three augmentations are available:
        - intensity variation
        """

        if cfg.add_noise and self.mode is self.MODES.TRAIN:
            if cfg.noise_typ == NOISETYPE.GAUSSIAN:
                gaussian = np.random.normal(0, cfg.standard_deviation, img.shape)
                logger.debug("Minimum Gauss %.3f:", gaussian.min())
                logger.debug("Maximum Gauss %.3f:", gaussian.max())
                img = img + gaussian

            elif cfg.noise_typ == NOISETYPE.POISSON:
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
        aug_spc = cfg.sample_target_spacing
        # see if any values in target spacing are none
        if np.any([ts is None for ts in aug_spc]):
            aug_target_spacing = []
            for num, spc in enumerate(aug_spc):
                if spc is None:
                    aug_target_spacing.append(image.GetSpacing()[num])
                else:
                    aug_target_spacing.append(spc * resolution_augmentation)
        else:
            aug_target_spacing = list(aug_spc)
        logger.debug("        Spacing %s", aug_target_spacing)

        size = np.array(image.GetSize())
        spacing = np.array(image.GetSpacing())
        new_size = [int(i) for i in size * spacing / aug_target_spacing]

        # resample the image
        resample_method = sitk.ResampleImageFilter()
        resample_method.SetOutputSpacing(aug_target_spacing)
        resample_method.SetDefaultPixelValue(0)
        resample_method.SetInterpolator(sitk.sitkLinear)
        resample_method.SetOutputDirection(image.GetDirection())
        resample_method.SetOutputOrigin(image.GetOrigin())
        resample_method.SetSize(new_size)
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

    def _get_class_reg(self, file_name):
        # convert to string if necessary
        if isinstance(file_name, bytes):
            file_id = str(file_name, "utf-8")
        else:
            file_id = str(file_name)

        if self.n_classification > 0 or self.n_regression > 0:
            class_samples = self.file_dict[file_id].get("classification", [])
            reg_samples = self.file_dict[file_id].get("regression", [])
            # make sure the regression samples have a shape
            reg_samples = [np.array(s).reshape((1,)) for s in reg_samples]
            file_samples = tuple(class_samples + reg_samples)
            # duplicate the samples self.samples_per_volume times
            file_samples_expanded = tuple(
                np.tile(s, (self.samples_per_volume,) + (1,) * s.ndim) for s in file_samples
            )
            return file_samples_expanded
        else:
            return tuple()

    def _get_autoencoder(self, file_name, sample_np):
        # convert to string if necessary
        if isinstance(file_name, bytes):
            file_id = str(file_name, "utf-8")
        else:
            file_id = str(file_name)
        if "autoencoder" in self.tasks:
            output_type = self.file_dict[file_id]["autoencoder"]
            if output_type == "image":
                return (sample_np,)
            else:
                raise ValueError(f"Output type {output_type} unknown.")
        else:
            return tuple()
