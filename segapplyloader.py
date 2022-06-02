"""This model can be used to load images into tensorflow. The segbasisloader
will augment the images while the apply loader can be used to pass whole images.
"""
import logging
import os
from typing import Dict

import numpy as np
import SimpleITK as sitk

from . import config as cfg
from .segbasisloader import SegBasisLoader

# configure logger
logger = logging.getLogger(__name__)


class ApplyBasisLoader(SegBasisLoader):
    """The loader to apply the data to an image. It will mainly just preprocess
    and pad the image and return the values if called, it will not be converted
    to a data loader and should just be used for single images.

    There are also a few functions which can be used to make the loading
    easier, caching is used and the loader can also remove the added padding.

    Parameters
    ----------
    file_dict : Dict[str, Dict[str, str]]
        dictionary containing the file information, the key should be the id of
        the data point and value should be a dict with the image and labels as
        keys and the file paths as values
    mode : has no effect, should be None or APPLY
    seed : int, optional
        Has no effect, by default 42
    name : str, optional
        The name, by default 'apply_loader'
    """

    def __init__(
        self,
        file_dict: Dict[str, Dict[str, str]],
        mode=None,
        seed=42,
        name="apply_loader",
        divisible_by=16,
        **kwargs,
    ):
        if mode is None:
            mode = self.MODES.APPLY
        assert mode == self.MODES.APPLY, "Use this loader only to apply data to an image"

        super().__init__(file_dict=file_dict, mode=mode, seed=seed, name=name, **kwargs)

        if len(cfg.train_input_shape) == 4:
            self.training_shape = np.array(cfg.train_input_shape)
        elif len(cfg.train_input_shape) == 3:
            # add z-dimension of 1
            self.training_shape = np.array([1] + cfg.train_input_shape)
        else:
            raise ValueError("cfg.train_input_shape should have 3 or 4 dimensions")
        # do not use caching when applying
        self.use_caching = False
        self.label = False

        # have all sizes except the channel divisible by a certain number (needed for downsampling)
        self.divisible_by = divisible_by
        assert self.divisible_by % 2 == 0
        assert isinstance(self.divisible_by, int)

        # remember the last file
        self.last_file = None
        self.last_file_name = None
        self.last_indices = None
        self.last_stride = None
        self.last_window_shape = None
        self.last_overlap = None
        self.last_shape = None
        self.last_padding = None

    # if called with filename, just return the padded test sample
    def __call__(self, filename, *args, **kwargs):
        """Returns the padded  and preprocessed test sample

        Parameters
        ----------
        filename : str
            The filename, if it does not exist, it will be converted using the
            framework format. If it is already processed, that will be used

        Returns
        -------
        np.array
            the padded sample
        """
        return self.get_padded_test_sample(filename)

    def _set_up_shapes_and_types(self):
        """!
        sets all important configurations from the config file:
        - n_channels
        - dtypes
        - dshapes
        - slice_shift

        also derives:
        - data_rank
        - slice_shift

        """
        # pylint: disable=attribute-defined-outside-init
        self.dtypes = [cfg.dtype]
        self.data_rank = len(cfg.train_input_shape)
        self.dshapes = []

    def _get_samples_from_volume(self, data_img, label_img=None):

        # convert samples to numpy arrays
        data = sitk.GetArrayFromImage(data_img)
        # add 4th dimension if it is not there
        if data.ndim == 3:
            data = np.expand_dims(data, axis=-1)

        if self.mode != self.MODES.APPLY:
            raise NotImplementedError("Use this loader only to apply data to an image")

        # if rank is 4, add batch dimension
        if self.data_rank == 4:
            data = np.expand_dims(data, axis=0)
        # TODO: remove, add that in the apply code

        # set shape
        self.dshapes = [data.shape[1:]]  # pylint: disable=attribute-defined-outside-init
        if not self.data_rank == len(self.dshapes[0]):
            raise ValueError(f"Data has rank {len(self.dshapes[0])}, not {self.data_rank}.")

        return [data, None]

    def _load_file(self, file_name, load_labels=True, **kwargs):
        # convert to string if necessary
        if isinstance(file_name, bytes):
            file_name = str(file_name, "utf-8")
        else:
            file_name = str(file_name)

        # see if there is a saved file
        if self.last_file is not None:
            # if the file name is the same
            if self.last_file_name == file_name:
                return self.last_file, None

        # see if filename should be converted
        if os.path.exists(file_name):
            logger.debug("        Loading %s (%s)", file_name, self.mode)
            # load image
            data_img = sitk.ReadImage(file_name)

            return data_img, None

        # if it does not exist, dataset conversion will be tried
        else:
            # attempt to convert it
            file_name_converted, _ = self.get_filenames(file_name)
            # if that also does not work, raise error
            if not os.path.exists(file_name_converted):
                raise FileNotFoundError(f"The test file {file_name} could not be found.")
            # otherwise, load it
            data, _ = super()._load_file(file_name, load_labels=False)
            return data, None

    def get_test_sample(self, filename):
        """Get the preprocessed test sample without padding

        Parameters
        ----------
        filename : str
            The filename

        Returns
        -------
        np.array
            the test file
        """
        data, _ = self._load_file(filename)
        return self._get_samples_from_volume(data)[0]

    def get_padded_test_sample(self, filename, min_padding=None) -> np.ndarray:
        """Get an image, preprocess and pad it

        Parameters
        ----------
        filename : str
            The name of the file to load
        min_padding : int, optional
            The minimum amount of padding to use, if None, 0 will be used or the
            amount needed to pad up to the training shape, by default None

        Returns
        -------
        np.array
            The padded image as array
        """
        data = self.get_test_sample(filename)

        assert self.data_rank is not None, "Set up shapes and types first"
        pad_with = np.zeros((1 + self.data_rank, 2), dtype=int)
        # do not pad the batch axis (z for the 2D case) and the last axis (channels)
        min_dim = 1

        if min_padding is None:
            # make sure to round up
            min_p = 0
            # see if data is smaller than the training shape
            min_index = 4 - self.data_rank
            max_diff = (self.training_shape[min_index:3] - data.shape[min_index:3]).max()
            min_p = np.maximum(max_diff, min_p)
        else:
            min_p = min_padding

        # make sure the patches are divisible by a certain number
        div_h = self.divisible_by // 2
        for num, size in zip(range(min_dim, 4), data.shape[min_dim:-1]):
            if size % 2 == 0:
                # and make sure have of the final size is divisible by divisible_by
                if div_h == 0:
                    pad_div = 0
                else:
                    pad_div = div_h - ((size // 2 + min_p) % div_h)
                pad_with[num] = min_p + pad_div
            else:
                # and make sure have of the final size is divisible by divisible_by
                if div_h == 0:
                    pad_div = 0
                else:
                    pad_div = div_h - (((size + 1) // 2 + min_p) % div_h)
                pad = min_p + pad_div
                # pad asymmetrical
                pad_with[num, 0] = pad + 1
                pad_with[num, 1] = pad

        # pad the data (using 0s)
        data_padded = np.pad(data, pad_with)

        # remember padding
        self.last_padding = pad_with
        self.last_shape = data_padded.shape

        return data_padded

    def remove_padding(self, data: np.ndarray):
        """Remove the padding, shape has to be the same as the test image.

        Parameters
        ----------
        data : np.array
            The padded output data

        Returns
        -------
        np.array
            The output without padding
        """
        padding = self.last_padding
        assert self.last_shape is not None, "only remove padding after adding it."
        assert (
            data.shape[:3] == self.last_shape[:3]
        ), "data shape does not match the padding"

        if self.data_rank == 3:
            assert padding.shape == (4, 2)
        if self.data_rank == 4:
            assert padding.shape == (5, 2)

        # cut off the padding
        for num, (first, last) in enumerate(padding):
            data = np.take(data, indices=np.arange(first, data.shape[num] - last), axis=num)

        return data

    def get_windowed_test_sample(self, img, window_shape, overlap=None):
        """If images are too big, this returns a padded, windowed view of the image

        Parameters
        ----------
        img : np.array
            The padded, preprocessed image to window
        window_shape : something that can be turned into a numpy array
            The shape of the window to use with the extent of the window (z,y,x)
            if only one number is provided, this is used as z-extent
        overlap : int
            The overlap between two windows, if None, 15 will be used, default None
            For 2D data, there will only be overlap in-plane.

        Returns
        -------
        np.array
            The windowed, padded view of the array with shape (n_patches, 1) +  window_shape + (n_channels,)
        """

        # get the overlap (this is derived from the training shape and used as padding between slices)
        if overlap is None:
            if self.data_rank == 3:
                overlap = [0] + [15] * 2
            else:
                overlap = [15] * 3
        elif np.issubdtype(type(overlap), int):
            if self.data_rank == 3:
                overlap = [0] + [overlap] * 3
            else:
                overlap = [overlap] * 3
        else:
            assert len(overlap) == 3, "Overlap should have length 3"
        self.last_overlap = overlap

        assert np.all(img.shape == self.last_shape), "Shape does not match the last image"

        # remove the batch dimension
        if img.ndim == 5:
            img = img[0]

        if isinstance(window_shape, int):
            window_shape = (window_shape, img.shape[1], img.shape[2])
        window_shape = np.array(window_shape)
        assert len(window_shape) == 3, "Size should have three entries"

        # the window should be smaller than the image
        window_shape = np.min([window_shape, img.shape[:3]], axis=0)

        # and larger than the training shape
        assert np.all(window_shape >= self.training_shape[:3]), (
            "The window_shape should be bigger "
            + f"than the training shape {self.training_shape[:3]}"
        )

        # calculate the stride
        stride = np.zeros(3, dtype=int)
        for i in range(3):
            if window_shape[i] < img.shape[i]:
                # the stride uses two times the overlap, because it is needed for both patches
                stride[i] = window_shape[i] - overlap[i] * 2
            else:
                # in this case, window and image shape are the same, the stride should be the image shape
                stride[i] = 1

        # add channel dimension to the window shape
        window_shape_with_ch = np.concatenate([window_shape, [img.shape[3]]])

        # remember window shape
        self.last_window_shape = window_shape
        # remember stride
        self.last_stride = stride

        # Sliding window view of the array. The sliding window dimensions are inserted at the end,
        # and the original dimensions are trimmed as required by the size of the sliding window.
        # That is, view.shape = x_shape_trimmed + window_shape, where x_shape_trimmed is x.shape
        # with every entry reduced by one less than the corresponding window size.
        sliding_view = np.lib.stride_tricks.sliding_window_view(
            img, window_shape_with_ch, axis=(0, 1, 2, 3), writeable=False
        )

        # for the indices, use one more step than there would be in the windowed image
        max_indices = np.zeros(3, dtype=int)
        for i in range(3):
            # if the stride fits into the shape, do nothing
            if sliding_view.shape[i] % stride[i] == 0:
                max_indices[i] = sliding_view.shape[i]
            # if not, add one more step
            else:
                max_indices[i] = (sliding_view.shape[i] // stride[i] + 1) * stride[i] + 1
        indices = np.indices(max_indices)
        # use the stride on the images
        indices_stride = indices[:, :: stride[0], :: stride[1], :: stride[2]]
        # clip to the maximum index, this ensures that also the edge is covered
        # reshape
        indices_clipped = indices_stride.reshape((3, -1)).T
        for i in range(3):
            indices_clipped[:, i] = np.clip(
                indices_clipped[:, i], 0, sliding_view.shape[i] - 1
            )

        # remember them
        self.last_indices = indices_clipped

        # apply the indices to the sliding_view
        patches = np.zeros((indices_clipped.shape[0], 1) + tuple(window_shape_with_ch))
        for num, idx in enumerate(indices_clipped):
            # assign the patches (with 0 for the channel dimension)
            patches[num] = sliding_view[idx[0], idx[1], idx[2], 0]

        return patches

    def stitch_patches(self, patches) -> np.ndarray:
        """Stitch the patches together to get one image.

        Parameters
        ----------
        patches : list
            List of patches to stitch in the same order as from the
            get_windowed_test_sample function.

        Returns
        -------
        np.ndarray
            The stitched image
        """
        assert self.last_window_shape is not None, "only stitch patches after creating them"
        patches = np.array(patches)
        if self.data_rank == 3:
            assert len(patches.shape) == 4, "dimensions should be 4"
            assert np.all(
                patches.shape[1:3] == self.last_window_shape[1:]
            ), "wrong patch shape"
        else:
            assert len(patches.shape) == 6, "dimensions should be 6"
            assert patches.shape[1] == 1, "batch number should be 1"
            assert np.all(patches.shape[2:5] == self.last_window_shape), "wrong patch shape"
        assert patches.shape[0] == self.last_indices.shape[0], "wrong number of patches"
        # last dimension is unknown, it is the number of channels in the input
        # and the number of classes in the output

        # use the shape of the last image, except the number of channels, which is replaced by the number of classes
        stitched_image = np.zeros(self.last_shape[:-1] + (patches.shape[-1],))
        ovl = self.last_overlap
        for num, indices in enumerate(self.last_indices):
            if self.data_rank == 3:
                stitched_image[
                    indices[0] + ovl[0] : indices[0] - ovl[0] + self.last_window_shape[0],
                    indices[1] + ovl[1] : indices[1] - ovl[1] + self.last_window_shape[1],
                    indices[2] + ovl[2] : indices[2] - ovl[2] + self.last_window_shape[2],
                    :,  # channel stay the same
                ] = patches[num, ovl[1] : -ovl[1], ovl[2] : -ovl[2], :]
            else:
                stitched_image[
                    :,  # batch stays the same
                    indices[0] + ovl[0] : indices[0] - ovl[0] + self.last_window_shape[0],
                    indices[1] + ovl[1] : indices[1] - ovl[1] + self.last_window_shape[1],
                    indices[2] + ovl[2] : indices[2] - ovl[2] + self.last_window_shape[2],
                    :,  # channel stay the same
                ] = patches[
                    num, :, ovl[0] : -ovl[0], ovl[1] : -ovl[1], ovl[2] : -ovl[2], :
                ]  # pylint: disable=invalid-unary-operand-type
        return stitched_image

    def get_original_image(self, filename):
        """Get the original image, without any preprocessing, this can be saved
        somewhere else or as a reference for resampling

        Parameters
        ----------
        filename : str
            The filename

        Returns
        -------
        sitk.Image
            The original image
        """
        img, _ = self.get_filenames(filename)
        return sitk.ReadImage(img)

    def get_processed_image(self, filename):
        """Get the preprocessed image

        Parameters
        ----------
        filename : str
            The filename

        Returns
        -------
        sitk.Image
            The preprocessed image
        """
        data, _ = self._load_file(filename)
        return data
