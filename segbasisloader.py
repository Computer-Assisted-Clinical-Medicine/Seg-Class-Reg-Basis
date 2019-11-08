import numpy as np
np.seterr(all='raise')
import SimpleITK as sitk
import os
from . import config as cfg
from .NetworkBasis import image as Image
from .NetworkBasis.dataloader import DataLoader


class SegBasisLoader(DataLoader):
    """!
    LitsLoader Class
    """

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
            self.dshapes = [cfg.train_input_shape, cfg.train_label_shape]
        else:
            self.dtypes = [cfg.dtype]
            self.dshapes = [cfg.test_data_shape]

        self.data_rank = len(self.dshapes[0])

        # Automatically determin data dimension
        if self.data_rank == 3:
            print('    Rank of input shape is', self.data_rank, '. Loading 2D samples.')
            # In 2D this parameter describes how many slices are additionally loaded.
            # They will be interpreted as channels.
            # Using in_between_slice_factor slices will be skipped.
            self.slice_shift = ((self.n_channels - 1) // 2) * cfg.in_between_slice_factor
        elif self.data_rank == 4:
            print('    Rank of input shape is', self.data_rank, '. Loading 3D samples.')
            # In 3D this parameter describes the z extend of teh sample.
            if self.mode is self.MODES.APPLY:
                print("Slice shift will we be set dynamically.")
            else:
                self.slice_shift = self.dshapes[0][0] // 2
                print(self.slice_shift)
        else:
            raise Exception('rank = \'{}\' is not supported'.format(self.data_rank))

    def _set_up_capacities(self):
        """!
        sets buffer size for sample buffer based on cfg.batch_capacity_train and
        cfg.batch_capacity_train based on self.mode

        """
        if self.mode is self.MODES.TRAIN:
            self.sample_buffer_size = cfg.batch_capacity_train
        elif self.mode is self.MODES.VALIDATE:
            self.sample_buffer_size = cfg.batch_capacity_train

    def _read_file_and_return_numpy_samples(self, file_id):
        """!
        Calls _load_file to load data and label based on the @p file_id and then extracts samples by calling _get_samples_from_volume()
        @param file_id <em>string,  </em> a file ID correspondinging to a pair of data file and label

        This function operates as follows:
        - calls _load_file()
        - calls _get_samples_from_volume()

        @return numpy arrays I (containing input samples) and if mode is not APPLY L (containing according labels)

        """
        # read and precrocess data, return image and label as 3D numpy matrices
        data, lbl = self._load_file(file_id)
        samples, labels = self._get_samples_from_volume(data, lbl)
        if self.mode is not self.MODES.APPLY:
            return samples, labels
        else:
            return [samples]

    def _load_file(self, file_id):
        '''!
        loads nii data file and nii labels file given the @p file_id

        @param file_id <em>string,  </em> should have the format <tt> 'Location\\file_number'</tt>. The @p Location must contain the data file with the @p file_number named as @p volume-file_number.nii and the data file with the @p file_number named as @p segmentation-file_number.nii to be loaded. For example, <tt>'C:\\DataLoction\\0'</tt> can be a @p file_id. Then <tt> C:\\DataLoction </tt> should contain @p volume-0.nii and @p segmetation-0.nii.

        Given the @p file_id, this function
        - Generates the location and file number for the data and label files.
        - Reads data and labels files using a SimpleITK reader.
        - Resamples (With augmentation if <tt>  self.mode </tt> is @p 'train'). See resample().
        - Combines liver and tumor annotations into one label.
        - Generates data and labels numpy arrays from nii files.
        - Moves z axis to last dimension.

        @return data and labels as numpy arrays
        '''
        file_id = str(file_id, 'utf-8')
        folder, file_number = os.path.split(file_id)
        print('        Loading ', folder, file_number, ' (', self.mode, ')')
        # Use a SimpleITK reader to load the nii images and labels for training
        data_img = sitk.ReadImage(os.path.join(folder, (cfg.sampe_file_name_prefix + file_number + '.nii')))
        label_img = sitk.ReadImage(os.path.join(folder, (cfg.label_file_name_prefix + file_number + '.nii')))
        data_img, label_img = self.adapt_to_task(data_img, label_img)
        data_img, label_img = self._resample(data_img, label_img)
        data = sitk.GetArrayFromImage(data_img)
        data = self.normalize(data)
        lbl = sitk.GetArrayFromImage(label_img)
        # move z axis to last index
        data = np.moveaxis(data, 0, -1)
        lbl = np.moveaxis(lbl, 0, -1)
        self._check_images(data, lbl)

        return data, lbl

    def _get_samples_from_volume(self, data, lbl):
        '''!
        Get samples for training from image and label data.

        @param data <em>numpy array,  </em> input image data with config.num_channels channels
        @param lbl <em>numpy array,  </em> label image, one channel

        This function operates as follows:
        - calls _select_indices()
        - calls _get_samples_by_index() for each index
        - calls _augment_samples() on the input samples

        @return numpy arrays I (containing input samples) and if mode is not APPLY L (containing according labels)
        '''
        indices, sampling_rates = self._select_indices(data, lbl)

        if self.mode is self.MODES.APPLY:
            if self.data_rank == 4:
                I = np.zeros((1, self.slice_shift * 2, *self.dshapes[0][1:]))
            else:
                I = np.zeros((len(indices), *self.dshapes[0]))

            for i in range(len(indices)):
                images, _ = self._get_samples_by_index(data, lbl, indices[i], samples_per_slice=1)
                I[i] = images

            print('   Image Samples Shape: ', I.shape)
            return [I, None]

        else:
            L = []
            I = []

            for i in range(len(indices)):
                images, labels = self._get_samples_by_index(data, lbl, indices[i], sampling_rates[i])
                I.extend(images)
                L.extend(labels)

            L = np.array(L)
            I = np.array(I)

            if self.mode is self.MODES.TRAIN:
                I, L = self._augment_samples(I, L)

            print('   Image Samples Shape: ', I.shape)
            print('   Label Samples Shape: ', L.shape)

            return [I, L]

    def _select_indices(self, data, lbl):
        '''!
        select slice indices from which samples should be taken
        # TODO: update description

        @param lbl <em>numpy array,  </em> label image, where background is zero

        Z-axis / craniocaudal axis is assumed to be third dimension. Across the third dimension the slices are differentiated
        into zero (empty slices/no segmentation) and non-zero slices.
        For <tt>  self.mode </tt> = @p 'train' all non-zero slices are added to the index list.
        For <tt>  self.mode </tt> = @p 'validate' the number of slices is restricted by config.validation_num_non_zero_indices.
        The number of zero samples is calculated in relation to the non-zero samples based on config.non_zero_sample_percent. Should this number
        be higher than the number actual number of zero indices, then only the available indices are added.

        @return slice indices as numpy array
        '''
        s = lbl.shape[2]  # third dimension is z-axis
        if self.data_rank == 4 and self.mode == self.MODES.APPLY:
            self.slice_shift = s // 2
            return [s // 2], np.ones(1, dtype=np.int)
        else:
            indices = np.arange(0 + self.slice_shift, s - self.slice_shift, 1)
            if self.mode == self.MODES.APPLY:
                return indices, np.ones(indices.size, dtype=np.int)
            else:
                return np.random.permutation(indices), np.ones(indices.size, dtype=np.int) * cfg.samples_per_slice_uni

    def _get_samples_by_index(self, data, label, index, samples_per_slice=cfg.samples_per_slice_uni, object_sampling=True):
        '''!
        Get samples from index slice.

        @param data <em>numpy array,  </em> input image
        @param label <em>numpy array,  </em> label image
        @param index <em>int,  </em> index of the slices that should be processed
        @param samples_per_slice <em>int,  </em> how many samples will be taken from this slice

        This is for Training.
        This function operates as follows:
        - based on cfg.random_sampling_mode patch centers are selected.
        - data and label are then extracted

        @return numpy arrays I (containing input samples) and L (containing according labels), if mode is APPLY L is None
        '''

        data_shape = data.shape
        if self.data_rank == 3:
            bb_dim = [self.dshapes[0][0], self.dshapes[0][1]]
        else:
            bb_dim = [self.dshapes[0][1], self.dshapes[0][2]]
        use_bb_boundaries = False
        slice_is_empty = False

        if self.mode is not self.MODES.APPLY:
            if object_sampling:
                if cfg.normalizing_method == cfg.NORMALIZING.WINDOW:
                    n_z = np.nonzero(label[:, :, index] * np.greater(data[:, :, index], -1))  # do not use air
                else:
                    n_z = np.nonzero(label[:, :, index])
            else:
                if cfg.normalizing_method == cfg.NORMALIZING.WINDOW:
                    n_z = np.where(
                        label[:, :, index] * np.greater(data[:, :, index], -1) == 0)  # do not use air
                else:
                    n_z = np.where(label[:, :, index] == 0)

            if cfg.random_sampling_mode == cfg.SAMPLINGMODES.CONSTRAINED_LABEL:
                # if there are no lables on the slice, sample uniformly for CONSTRAINED_LABEL
                use_bb_boundaries = True

            if n_z[0].size == 0:
                slice_is_empty = True
                if cfg.random_sampling_mode == cfg.SAMPLINGMODES.CONSTRAINED_MUSTD:
                    # if there are no lables on the slice, sample in the body for CONSTRAINED_MUSTD
                    # print('This is where it brakes.')
                    if cfg.normalizing_method == cfg.NORMALIZING.WINDOW:
                        n_z = np.nonzero(np.greater(data[:, :, index], cfg.norm_min_v))
                    elif cfg.normalizing_method == cfg.NORMALIZING.MEAN_STD:
                        n_z = np.nonzero(np.greater(data[:, :, index], 0))
                        # print(n_z)

            if cfg.random_sampling_mode == cfg.SAMPLINGMODES.CONSTRAINED_MUSTD:
                if n_z[0].size > 0:
                    mean_x = np.mean(n_z[0], dtype=np.int32)
                    std_x = np.std(n_z[0], dtype=np.int32) * cfg.patch_shift_factor
                    mean_y = np.mean(n_z[1], dtype=np.int32)
                    std_y = np.std(n_z[1], dtype=np.int32) * cfg.patch_shift_factor

                    # ensure that bounding box is inside data
                    min_x = min(max(mean_x - std_x, bb_dim[0] // 2), data_shape[0] - bb_dim[0] // 2)
                    max_x = max(min(mean_x + std_x + 1, data_shape[0] - bb_dim[0] // 2), bb_dim[0] // 2)
                    min_y = min(max(mean_y - std_y, bb_dim[1] // 2), data_shape[1] - bb_dim[1] // 2)
                    max_y = max(min(mean_y + std_y, data_shape[1] - bb_dim[1] // 2), bb_dim[1] // 2)
                else:
                    use_bb_boundaries = True

        if self.mode is self.MODES.APPLY or cfg.random_sampling_mode == cfg.SAMPLINGMODES.UNIFORM or use_bb_boundaries:
            min_x = bb_dim[0] // 2
            max_x = data_shape[0] - bb_dim[0] // 2
            min_y = bb_dim[1] // 2
            max_y = data_shape[1] - bb_dim[1] // 2

        if self.mode is self.MODES.APPLY and self.data_rank == 4:
            I = np.zeros((1, self.slice_shift * 2,  *self.dshapes[0][1:]))
        else:
            I = np.zeros((samples_per_slice, *self.dshapes[0]))

        if self.mode is not self.MODES.APPLY:
            L = np.zeros((samples_per_slice, *self.dshapes[1][:-1]))

        # select samples
        if cfg.random_sampling_mode == cfg.SAMPLINGMODES.UNIFORM or cfg.random_sampling_mode == cfg.SAMPLINGMODES.CONSTRAINED_MUSTD or slice_is_empty:
            sample_x = np.random.randint(min_x, max_x + 1, samples_per_slice)
            sample_y = np.random.randint(min_y, max_y + 1, samples_per_slice)

        elif cfg.random_sampling_mode == cfg.SAMPLINGMODES.CONSTRAINED_LABEL:
            # select patch centers from mask points inside the bounding box
            valid_locations = np.nonzero(np.logical_and(np.logical_and(n_z[0] >= min_x, n_z[0] <= max_x),
                                                        np.logical_and(n_z[1] >= min_y, n_z[1] <= max_y)))[0]
            # print('   VL:', valid_locations.shape, valid_locations)
            # If there are not enough centers, select voxels close by
            if len(valid_locations) < samples_per_slice:
                n_missing_samples = samples_per_slice - len(valid_locations)

                def find_nearest(array, value):
                    array = np.asarray(array)
                    dist = np.abs(array - value)
                    idx = dist.argmin()
                    distance = min(dist[idx], array[1] - array[0])
                    # don't touch this
                    distance = distance if distance > 1 else 1
                    return idx, distance

                x_idx, x_dist = find_nearest([min_x, max_x], np.mean(n_z[0]))
                y_idx, y_dist = find_nearest([min_y, max_y], np.mean(n_z[1]))

                # print('Locations and Distances: ', np.mean(n_z[0]), '->', x_idx, x_dist, [min_x, max_x], '|',
                #       np.mean(n_z[1]), '->', y_idx, y_dist, [min_y, max_y])

                if x_idx == 0:  # if minimum
                    sample_x = min_x + np.random.randint(0, x_dist + 1, n_missing_samples)
                else:
                    sample_x = max_x - np.random.randint(0, x_dist + 1, n_missing_samples)

                if y_idx == 0:
                    sample_y = min_y + np.random.randint(0, y_dist + 1, n_missing_samples)
                else:
                    sample_y = max_y - np.random.randint(0, y_dist + 1, n_missing_samples)

                sample_x = np.append(sample_x, n_z[0][valid_locations])
                sample_y = np.append(sample_y, n_z[1][valid_locations])

            # Otherwise select randomly
            else:
                selection = np.random.choice(valid_locations, samples_per_slice, replace=False)
                sample_x = n_z[0][selection]
                sample_y = n_z[1][selection]

        else:
            # TODO: throw error for not allowed mode
            pass

        # Get Sample Data from Volumes
        for i in range(samples_per_slice):
            if self.data_rank == 3:
                I[i] = data[sample_x[i] - (bb_dim[0] // 2):sample_x[i] + (bb_dim[0] // 2),
                       sample_y[i] - (bb_dim[1] // 2):sample_y[i] + (bb_dim[1] // 2),
                       index - self.slice_shift:index + self.slice_shift + 1:cfg.in_between_slice_factor]
            else:
                I[i] = np.expand_dims(np.moveaxis(data[sample_x[i] - (bb_dim[0] // 2):sample_x[i] + (bb_dim[0] // 2),
                                  sample_y[i] - (bb_dim[1] // 2):sample_y[i] + (bb_dim[1] // 2),
                                  index - self.slice_shift:index + self.slice_shift:1], -1, 0), -1)

            if self.mode is not self.MODES.APPLY:
                if self.data_rank == 3:
                    L[i] = label[sample_x[i] - (bb_dim[0] // 2): sample_x[i] + (bb_dim[0] // 2),
                           sample_y[i] - (bb_dim[1] // 2): sample_y[i] + (bb_dim[1] // 2),
                           index]
                else:
                    L[i] = np.moveaxis(label[sample_x[i] - (bb_dim[0] // 2): sample_x[i] + (bb_dim[0] // 2),
                                       sample_y[i] - (bb_dim[1] // 2): sample_y[i] + (bb_dim[1] // 2),
                                       index - self.slice_shift:index + self.slice_shift:1], -1, 0)

        if self.mode is not self.MODES.APPLY:
            L = self.__class__.one_hot_label(L)
            return [I, L]
        else:
            return [I, None]


    @staticmethod
    def one_hot_label(label):
        '''!
        convert a one-channel binary label image into a one-hot label image

        @param label <em>numpy array,  </em> label image, where background is zero and object is 1 (only works for binary problems)

        @return two-channel one-hot label as numpy array
        '''
        # add empty last dimension
        if label.shape[-1] > 1:
            label = np.expand_dims(label, axis=-1)

        if cfg.num_classes_seg == 2:
            # Todo: add empty first dimension if single sample
            # if label.ndim < 3:
            #     label = np.expand_dims(label, axis=0)

            invert_label = 1 - label  # complementary binary mask
            return np.concatenate([invert_label, label], axis=-1)  # fuse
        else:
            label_list = []
            for i in range(cfg.num_classes_seg):
                label_list.append(label == i)
            return np.concatenate(label_list, axis=-1)  # fuse

    def _augment_samples(self, I, L):
        '''!
        samplewise data augmentation

        @param I <em>numpy array,  </em> image samples
        @param L <em>numpy array,  </em> label samples

        Three augmentations are available:
        - flipping coronal
        - flipping sagittal
        - intensity variation
        '''
        if cfg.do_flip_coronal or cfg.do_flip_sagittal or cfg.do_variate_intensities:
            for sample in range(I.shape[0]):
                if cfg.do_flip_coronal:
                    if np.random.randint(0, 2) > 0:
                        I[sample] = np.flip(I[sample], 1)
                        L[sample] = np.flip(L[sample], 1)
                if cfg.do_flip_sagittal:
                    if np.random.randint(0, 2) > 0:
                        I[sample, :, :, :] = np.flip(I[sample, :, :, :], 0)
                        L[sample, :, :, :] = np.flip(L[sample, :, :, :], 0)
                if cfg.do_variate_intensities:
                        variation = (np.random.random_sample() * 2.0 * cfg.intensity_variation_interval) - cfg.intensity_variation_interval
                        I[sample] = I[sample] + variation
                        if cfg.normalizing_method == cfg.NORMALIZING.WINDOW:
                            flags = I[sample] < -1
                            I[sample][flags] = -1
                            flags = I[sample] > 1
                            I[sample][flags] = 1
        return I, L

    def _resample(self, data, label):
        '''!
        This function operates as follows:
        - extract image meta information and assigns it to label as well
        - augmentation is only on in training
        - calls the static function _resample()

        @param data <em>ITK image,  </em> patient image
        @param label <em>ITK image,  </em> label image, 0 is background class

        @return resampled data and label images
        '''
        data_info = Image.get_data_info(data)
        label.SetDirection(data_info['orig_direction'])
        label.SetOrigin(data_info['orig_origin'])
        label.SetSpacing(data_info['orig_spacing'])

        target_info = {}
        target_info['target_spacing'] = cfg.target_spacing
        target_info['target_direction'] = cfg.target_direction
        target_info['target_size'] = cfg.target_size
        target_info['target_type_image'] = cfg.target_type_image
        target_info['target_type_label'] = cfg.target_type_image

        if self.mode is self.MODES.TRAIN:
            do_augment = True
        else:
            do_augment = False

        return Image.resample_sitk_image(data, target_info, data_background_value=cfg.data_background_value,
                                         do_adapt_resolution=cfg.adapt_resolution, label=label,
                                         label_background_value=cfg.label_background_value, do_augment=do_augment,
                                         max_rotation_augment=cfg.max_rotation)

    ### Static Functions ###

    @staticmethod
    def normalize(img, eps=np.finfo(np.float).min):
        '''
        Truncates input to interval [config.norm_min_v, config.norm_max_v] an
         normalizes it to interval [-1, 1] when using WINDOW and to mean = 0 and std = 1 when MEAN_STD.
        '''
        # ToDo: normalize by percentile
        flags = img < cfg.norm_min_v
        img[flags] = cfg.norm_min_v
        flags = img > cfg.norm_max_v
        img[flags] = cfg.norm_max_v
        if cfg.normalizing_method == cfg.NORMALIZING.WINDOW:
            img = (img - cfg.norm_min_v) / (cfg.norm_max_v - cfg.norm_min_v + cfg.norm_eps)
            img = (img * 2) - 1
        elif cfg.normalizing_method == cfg.NORMALIZING.MEAN_STD:
            img = img - np.mean(img)

            std = np.std(img)
            img = img / (std if std != 0 else eps)

            # np.seterr(all='raise')  # To catch invalid value in std warning
            # try:
            #     std = np.std(img)
            #     img = img / (std if std != 0 else eps)
            # except FloatingPointError as err:
            #     img = img / eps
            #     print(err, 'Replaced it with eps.')
        return img

    @staticmethod
    def normalize_sitk_image(img_data):
        """
        This method calls the normalize method on SimpleITK Images.
        It firstly extracts the data from SitkImage class,
        secondly applies the normalization and
        thirdly rebuilds the Data as SitkImage class and copies the image information (spacing and so on)

        :param img_data: image data to be normalized as SimpleITK.image
        :return: SimpleITK.image
        """
        img = sitk.GetArrayFromImage(img_data)
        img = SegBasisLoader.normalize(img)
        tmp_image_data = sitk.GetImageFromArray(img)
        tmp_image_data.CopyInformation(img_data)
        return tmp_image_data


