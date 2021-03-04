from enum import Enum
import logging
import os

import numpy as np
import SimpleITK as sitk

from . import config as cfg
from .NetworkBasis.dataloader import DataLoader
from .NetworkBasis import image

#configure logger
logger = logging.getLogger(__name__)

# define enums

class NORMALIZING(Enum):
    WINDOW = 0
    MEAN_STD = 1
    PERCENT5 = 2


class NOISETYP(Enum):
    GAUSSIAN =0
    POISSON = 1

class SegBasisLoader(DataLoader):

    def __init__(self, mode=None, seed=42, name='reader', frac_obj=None):
        super().__init__(mode=mode, seed=seed, name=name)

        # use caching
        self.use_caching = True

        # get the fraction of the samples that should contain the object
        if frac_obj == None:
            self.frac_obj = cfg.percent_of_object_samples / 100
        else:
            self.frac_obj = frac_obj

        self.normalizing_method = cfg.normalizing_method
        self.do_resampling = cfg.do_resampling

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
        self.n_channels = cfg.num_channels
        if self.mode is self.MODES.TRAIN or self.mode is self.MODES.VALIDATE:
            self.dtypes = [cfg.dtype, cfg.dtype]
            self.dshapes = [np.array(cfg.train_input_shape), np.array(cfg.train_label_shape)]
            # use the same shape for image and labels
            assert np.all(self.dshapes[0] == self.dshapes[1])
        else:
            self.dtypes = [cfg.dtype]
            self.dshapes = [np.array(cfg.test_data_shape)]

        self.data_rank = len(self.dshapes[0])

    def _set_up_capacities(self):
        """!
        sets buffer size for sample buffer based on cfg.batch_capacity_train and
        cfg.batch_capacity_train based on self.mode

        """
        if self.mode is self.MODES.TRAIN:
            self.sample_buffer_size = cfg.batch_capacity_train
        elif self.mode is self.MODES.VALIDATE:
            self.sample_buffer_size = cfg.batch_capacity_train

    def _get_filenames(self, file_id):
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
        s, _, l, _ = self._get_filenames_cached(file_id)
        return s, l

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

        # set preprocessing folder name
        folder_name = f'pre_{self.normalizing_method}'
        if self.do_resampling:
            folder_name += '_resampled'
        pre_folder = os.path.join(cfg.preprocessed_dir, folder_name)
        if not os.path.exists(pre_folder):
            os.makedirs(pre_folder)

        # generate the file name for the sample
        sample_name = cfg.sample_file_name_prefix + file_number
        data_file = os.path.join(folder, sample_name + cfg.file_suffix)
        if not os.path.exists(data_file):
            raise Exception(f'The file {data_file} could not be found')
        # generate the name of the preprocessed file
        filename = f'{sample_name}.mhd'
        data_file_pre = os.path.join(pre_folder, filename)

        # generate the file name for the sample
        label_name = cfg.label_file_name_prefix + file_number
        label_file = os.path.join(folder, (label_name + cfg.file_suffix))
        if not os.path.exists(data_file):
            raise Exception(f'The file {label_file} could not be found')
        # generate the name of the preprocessed file
        label_file_pre = os.path.join(pre_folder, (label_name + '.mhd'))
        return data_file, data_file_pre, label_file, label_file_pre

    def _load_file(self, file_name):
        # save preprocessed files as images, this increases the load time
        # from 20 ms to 50 ms per image but is not really relevant compared to
        #  the sampleing time. The advantage is that SimpleITK can be used for augmentation

        # convert to string if necessary
        if type(file_name) == bytes:
            file_id = str(file_name, 'utf-8')
        else:
            file_id = str(file_name)
        logger.debug('        Loading %s (%s)', file_id, self.mode)
        # Use a SimpleITK reader to load the nii images and labels for training
        data_file, data_file_pre, label_file, label_file_pre = self._get_filenames_cached(file_id) 
        # see if the preprocessed files exist
        if os.path.exists(data_file_pre) and os.path.exists(label_file_pre) and self.use_caching:
            # load images
            data = sitk.ReadImage(str(data_file_pre))
            lbl = sitk.ReadImage(str(label_file_pre))
        else:
            # load images
            data_img = sitk.ReadImage(data_file)
            label_img = sitk.ReadImage(label_file)
            # adapt, resample and normalize them
            data_img, label_img = self.adapt_to_task(data_img, label_img)
            if self.do_resampling:
                data_img, label_img = self._resample(data_img, label_img)
            data = sitk.GetArrayFromImage(data_img)
            data = self.normalize(data)
            lbl = sitk.GetArrayFromImage(label_img)
            # then check them
            self._check_images(data, lbl)
            # if everything is ok, save the preprocessed images
            data_img_proc = sitk.GetImageFromArray(data)
            data_img_proc.CopyInformation(data_img)
            label_img_proc = sitk.GetImageFromArray(lbl)
            label_img_proc.CopyInformation(label_img)
            sitk.WriteImage(data_img_proc, str(data_file_pre))
            sitk.WriteImage(label_img_proc, str(label_file_pre))
            data = data_img_proc
            lbl = label_img_proc

        return data, lbl

    def _check_images(self, data, labels):
        raise NotImplementedError('Should be implemented according to the task.')


    def normalize(self, img, eps=np.finfo(float).min):
        img_no_nan = np.nan_to_num(img, nan=cfg.data_background_value)
        # clip outliers and rescale to between zero and one
        a_min = np.quantile(img_no_nan, cfg.norm_min_q)
        a_max = np.quantile(img_no_nan, cfg.norm_max_q)
        if self.normalizing_method == NORMALIZING.PERCENT5:
            img = np.clip(img_no_nan, a_min=a_min, a_max=a_max)
            img = (img - a_min) / (a_max - a_min)
            img = (img * 2) - 1
        elif self.normalizing_method == NORMALIZING.MEAN_STD:
            img = np.clip(img_no_nan, a_min=a_min, a_max=a_max)
            img = img - np.mean(img)
            std = np.std(img)
            img = img / (std if std != 0 else eps)
        elif self.normalizing_method == NORMALIZING.WINDOW:
            img = np.clip(img_no_nan, a_min=cfg.norm_min_v, a_max=cfg.norm_max_v)
            img = (img - cfg.norm_min_v) / (cfg.norm_max_v - cfg.norm_min_v + eps)
            img = (img * 2) - 1
        else:
            raise NotImplementedError(f'{self.normalizing_method} is not implemented')

        return img

    def _resample(self, data, label=None):
        '''!
        This function operates as follows:
        - extract image meta information and assigns it to label as well
        - calls the static function _resample()

        @param data <em>ITK image,  </em> patient image
        @param label <em>ITK image,  </em> label image, 0 is background class

        @return resampled data and label images
        '''
        data_info = image.get_data_info(data)
        # assure that the images are similar
        if label is not None:
            assert np.allclose(data_info['orig_direction'], label.GetDirection(), atol=0.01), 'label and image do not have the same direction'
            assert np.allclose(data_info['orig_origin'], label.GetOrigin(), atol=0.01), 'label and image do not have the same origin'
            assert np.allclose(data_info['orig_spacing'], label.GetSpacing(), atol=0.01), 'label and image do not have the same spacing'

        target_info = {}
        target_info['target_spacing'] = cfg.target_spacing

        # make sure the spacing is orthogonal
        orig_dirs = np.array(data_info['orig_direction']).reshape(3,3)
        assert np.isclose(np.dot(orig_dirs[0],orig_dirs[1]), 0, atol=0.01), 'x and y not orthogonal'
        assert np.isclose(np.dot(orig_dirs[0],orig_dirs[2]), 0, atol=0.01), 'x and z not orthogonal'
        assert np.isclose(np.dot(orig_dirs[1],orig_dirs[2]), 0, atol=0.01), 'y and z not orthogonal'
        target_info['target_direction'] = data_info['orig_direction']

        # calculate the new size
        orig_size = np.array(data_info['orig_size']) * np.array(data_info['orig_spacing'])
        target_info['target_size'] = list((orig_size / np.array(cfg.target_spacing)).astype(int))

        target_info['target_type_image'] = cfg.target_type_image
        target_info['target_type_label'] = cfg.target_type_label

        return image.resample_sitk_image(data, target_info, data_background_value=cfg.data_background_value,
                                         do_adapt_resolution=self.do_resampling, label=label,
                                         label_background_value=cfg.label_background_value, do_augment=False)


    def _read_file_and_return_numpy_samples(self, file_name_queue:bytes):
        data_img, label_img = self._load_file(file_name_queue)
        samples, labels = self._get_samples_from_volume(data_img, label_img)
        if self.mode is not self.MODES.APPLY:
            return samples, labels
        else:
            return [samples]
        

    def _get_samples_from_volume(self, data_img, label_img):

        # augment whole images
        data_img, label_img = self._augment_images(data_img, label_img)

        # convert samples to numpy arrays
        data = sitk.GetArrayFromImage(data_img)
        lbl = sitk.GetArrayFromImage(label_img)

        # augment the numpy arrays
        data, lbl = self._augment_numpy(data, lbl)

        # check that there are labels
        assert np.any(lbl != 0), 'no labels found'
        # check shape
        assert np.all(data.shape[:-1] == lbl.shape)
        assert len(data.shape) == 4, 'data should be 4d'
        assert len(lbl.shape) == 3, 'labels should be 3d'
        
        # determine the number of background and foreground samples
        n_foreground = int(cfg.samples_per_volume * self.frac_obj)
        n_background = int(cfg.samples_per_volume * (1 - self.frac_obj))

        # calculate the maximum padding, so that at least three quarters in 
        # each dimension is inside the image
        # sample shape is without the number of channels
        if self.data_rank == 4:
            sample_shape = self.dshapes[0][:-1]
        # if the rank is three, add a dimension for the z-extent
        elif self.data_rank == 3:
            sample_shape = np.array([1,]+list(self.dshapes[0][:2]))
        assert sample_shape.size == len(data.shape)-1, 'sample dims do not match data dims'
        max_padding = sample_shape // 4

        # pad the data (using 0s)
        pad_with = ((max_padding[0],)*2, (max_padding[1],)*2, (max_padding[2],)*2)
        data_padded = np.pad(data, pad_with + ((0, 0),))
        label_padded = np.pad(lbl, pad_with)

        # calculate the allowed indices
        # the indices are applied to the padded data, so the minimum is 0
        # the last dimension, which is the number of channels is ignored
        min_index = np.zeros(3, dtype=int)
        # the maximum is the new data shape minus the sample shape (accounting for the padding)
        max_index = data_padded.shape[:-1] - sample_shape 
        assert np.all(min_index <= max_index), 'image to small to get patches'

        # get the background origins
        background_shape = (n_background, 3)
        origins_background = np.random.randint(low=min_index, high=max_index, size=background_shape)

        # get the foreground center
        valid_centers = np.argwhere(lbl)
        indices = np.random.randint(low=0, high=valid_centers.shape[0], size=n_foreground)
        origins_foreground = valid_centers[indices] + max_padding - sample_shape // 2
        # check that they are below the maximum amount of padding
        for i, m in enumerate(max_index):
            origins_foreground[:,i] = np.clip(origins_foreground[:,i], 0, m)

        # extract patches (pad if necessary), in separate function, do augmentation beforehand or with patches
        origins = np.concatenate([origins_foreground, origins_background])
        batch_shape = (n_foreground+n_background,) + tuple(sample_shape)
        samples = np.zeros(batch_shape + (self.n_channels,), dtype=cfg.dtype_np)
        labels = np.zeros(batch_shape, dtype=np.uint8)
        for num, (i,j,k) in enumerate(origins):
            sample_patch = data_padded[i:i+sample_shape[0],j:j+sample_shape[1],k:k+sample_shape[2]]
            label_patch = label_padded[i:i+sample_shape[0],j:j+sample_shape[1],k:k+sample_shape[2]]
            samples[num] = sample_patch
            labels[num] = label_patch
            # if num < n_foreground: # only for debugging
            #     assert np.sum(label_patch) > 0

        if self.mode == self.MODES.APPLY:
            raise NotImplementedError('Use the original data loader')

        # if rank is 3, squash the z-axes
        if self.data_rank == 3:
            samples = samples.squeeze(axis=1)
            labels = labels.squeeze(axis=1)

        # assert np.sum(labels) > 0 # only for debugging

        # augment and convert to one_hot_label
        if self.mode is not self.MODES.APPLY:
            # augment
            labels_onehot = np.squeeze(np.eye(cfg.num_classes_seg)[labels.flat]).reshape(labels.shape + (-1,))

        return samples, labels_onehot


    # TODO: clean up and document
    def _augment_numpy(self, I, L):
        '''!
        samplewise data augmentation

        @param I <em>numpy array,  </em> image samples
        @param L <em>numpy array,  </em> label samples

        Three augmentations are available:
        - flipping coronal
        - flipping sagittal
        - intensity variation
        - deformation
        '''

        if cfg.add_noise and self.mode is self.MODES.TRAIN:
            if cfg.noise_typ == NOISETYP.GAUSSIAN:
                gaussian = np.random.normal(0, cfg.standard_deviation, I.shape)
                logger.debug(f'Minimum Gauss{gaussian.min():.3f}:')
                logger.debug(f'Maximum Gauss {gaussian.max():.3f}:')
                I = I + gaussian

            elif cfg.noise_typ == NOISETYP.POISSON:
                poisson = np.random.poisson(cfg.mean_poisson, I.shape)
                # scale according to the values
                poisson = poisson * -cfg.mean_poisson/(cfg.norm_max_v - cfg.norm_min_v)
                logger.debug(f'Minimum Poisson {poisson.min():.3f}:')
                logger.debug(f'Maximum Poisson {poisson.max():.3f}:')
                I = I + poisson

        return I, L

    def _augment_images(self, data, label):
        '''!
        This function operates as follows:
        - extract image meta information and assigns it to label as well
        - augmentation is only on in training
        - calls the static function _resample()

        @param data <em>ITK image,  </em> patient image
        @param label <em>ITK image,  </em> label image, 0 is background class

        @return resampled data and label images
        '''
        data_info = image.get_data_info(data)

        target_info = {}
        target_info['target_spacing'] = data_info['orig_spacing']
        target_info['target_direction'] = data_info['orig_direction']

        # do not change the resolution
        target_info['target_size'] = data_info['orig_size']
        target_info['target_type_image'] = cfg.target_type_image
        target_info['target_type_label'] = cfg.target_type_label

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
                max_resolution_augment=cfg.max_resolution_augment
            )
        else:
            return data, label

class ApplyBasisLoader(SegBasisLoader):
    def __init__(self, mode=None, seed=42, name='apply_loader'):
        if mode is None:
            mode = self.MODES.APPLY
        assert(mode == self.MODES.APPLY), 'Use this loader only to apply data to an image'
        super().__init__(mode=mode, seed=seed, name=name)
        self.training_shape = np.array(cfg.train_input_shape)
        # do not use caching when applying
        self.use_caching = False
        self.label = False

    # if called with filename, just return the padded test sample
    def __call__(self, filename, *args, **kwargs):
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
        self.n_channels = cfg.num_channels
        self.dtypes = [cfg.dtype]
        self.data_rank = len(cfg.train_input_shape)
        self.dshapes = []


    def _get_samples_from_volume(self, data_img, label_img=None):

        # convert samples to numpy arrays
        data = sitk.GetArrayFromImage(data_img)

        if self.mode != self.MODES.APPLY:
            raise NotImplementedError('Use this loader only to apply data to an image')

        # if rank is 4, add batch dimension
        if self.data_rank == 4:
            data = np.expand_dims(data, axis=0)

        # set shape
        self.dshapes = [data.shape[1:]]
        self.rank = len(self.dshapes[0])

        return [data, None]

    def _load_file(self, file_name):
        # convert to string if necessary
        if type(file_name) == bytes:
            file_name = str(file_name, 'utf-8')
        else:
            file_name = str(file_name)

        # see if there is a saved file
        if hasattr(self, 'last_file'):
            # if the file name is the same
            if self.last_file_name == file_name:
                return self.last_file, None

        # see if filename should be converted
        if os.path.exists(file_name):
            logger.debug('        Loading %s (%s)', file_name, self.mode)
            # load image
            data_img = sitk.ReadImage(file_name)
            # adapt, resample and normalize them
            data_img, _ = self.adapt_to_task(data_img, None)
            if self.do_resampling:
                data_img = self._resample(data_img)
            data = sitk.GetArrayFromImage(data_img)
            data = self.normalize(data)
            # if everything is ok, save the preprocessed images
            data_img_proc = sitk.GetImageFromArray(data)
            data_img_proc.CopyInformation(data_img)

            # cache it in memory
            self.last_file = data_img_proc
            self.last_file_name = file_name

            return data_img_proc, None

        # if it does not exist, dataset conversion will be tried 
        else:
            # attemp to convert it
            file_name_converted, _ = self._get_filenames(file_name)
            # if that also does not work, raise error
            if not os.path.exists(file_name_converted):
                raise FileNotFoundError(f'The test file {file_name} could not be found.')
            # otherwise, load it
            data, _ = super()._load_file(file_name)
            return data, None

    def get_test_sample(self, filename):
        data, _ = self._load_file(filename)
        return self._get_samples_from_volume(data)[0]

    def get_padded_test_sample(self, filename):
        data = self.get_test_sample(filename)

        pad_with = np.zeros((1 + self.data_rank, 2), dtype=int)
        # do not pad the batch axis (z for the 2D case) and the last axis (channels)
        min_dim = 1
        for num, size in zip(range(min_dim, 4), data.shape[min_dim:-1]):
            # calculate the minimum padding (for rank==3, there is no z-dim)
            min_p = self.training_shape[num-min_dim] // 2
            if size % 2 == 0:
                # and make sure have of the final size is divisible by 16
                pad_with[num] = min_p + 8 - ((size // 2 + min_p) % 8)
            else:
                # and make sure have of the final size is divisible by 16
                p = min_p + 8 - (((size + 1) // 2 + min_p) % 8)
                # pad asymmetrical
                pad_with[num, 0] = p + 1
                pad_with[num, 1] = p            
        
        # pad the data (using 0s)
        data_padded = np.pad(data, pad_with)

        # remember padding
        self.last_padding = pad_with
        self.last_shape = data_padded.shape

        return data_padded

    def remove_padding(self, data):
        padding = self.last_padding
        assert data.shape==self.last_shape, 'data shape does not match the padding'

        if self.data_rank == 3:
            assert padding.shape == (4, 2)
        if self.data_rank == 4:
            assert padding.shape == (5, 2)

        # cut off the padding
        for num, (first, last) in enumerate(padding):
            data = np.take(
                data,
                indices=np.arange(first, data.shape[num]-last),
                axis=num
            )

        return data

    def get_original_image(self, filename):
        img, _ = self._get_filenames(filename)
        return sitk.ReadImage(img)

    def get_processed_image(self, filename):
        data, _ = self._load_file(filename)
        return data