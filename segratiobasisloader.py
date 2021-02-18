import logging
import traceback
import warnings

import numpy as np
import scipy
import tensorflow as tf

from . import config as cfg
from .segbasisloader import SegBasisLoader

#configure logger
logger = logging.getLogger(__name__)

class SegRatioBasisLoader(SegBasisLoader):

    def __call__(self, file_list, batch_size, n_epochs=1, read_threads=1):
        """!
        Parses the list of file IDs given in @p file_list, identifies .nrrd files to load and loads the data and labels, genetares @p tf.dataset of data and labels with necessary repeating, shuffling, batching and prefetching.

        @param  file_list : <em>  array of strings,  </em> where each string is a file ID corresponding to a pair of data file and label file to be loaded. @p file_list should be obtained from a @p .csv file and then converted to numpy array. Each ID string should have the format <tt> 'Location\\file_number'</tt>. From @p Location, the data file and label file with the @p file_number, respectiely named as @p volume-file_number.nrrd and segmentation-file_number.nrrd are loaded. (See also LitsLoader._read_file(), LitsLoader._load_file() for more details.)
        @param batch_size : <em> int, </em> batch size
        @param n_epochs : @em int, number of training epochs
        @param read_threads : @em int, number of threads/instances to read in parallel

        @return tf.dataset of data and labels

        This function operates as follows,
        - Generates Tensor of strings from the @p file_list
        - Creates @p file_list_ds dataset from the Tensor of strings.
        - If loader is in training mode (<tt>self.mode == 'train'</tt>),
            - @p file_list_ds is repeated with shuffle @p n_epoch times
            - @p file_list_ds is mapped with _read_wrapper() to obtain @p ds dataset. The mapping generates a set of samples from each pair of data and label files idenfied by each file ID. Here each element of dataset is a set of samples corresponding to one ID.
            - @p ds is flat mapped to make each element correspond to one sample inside the dataset.
            - @p ds is shuffles
            - @p ds is batched with @p batch_size
            - 1 element of @p ds is prefetched
            - @p ds is returned

        - Else if loader is in validation mode (<tt>self.mode == 'validation'</tt>),
            - @p file_list_ds is mapped with _read_wrapper() to obtain @p ds dataset (mapping is same as train mode)
            - @p ds is flat mapped to make each element correspond to one sample inside the dataset.
            - @p ds is batched with @p batch_size
            - @p ds is returned

        """
        if self.mode is self.MODES.APPLY:
            raise ValueError('Segmentation Ratio Loader is not applicable for Application, please use the regular Loader.')
        else:
            self.n_files = len(file_list)

            def concat(*ds_elements):
                # Create one empty list for each component of the dataset
                lists = [[] for _ in ds_elements[0]]
                for element in ds_elements:
                    for i, tensor in enumerate(element):
                        # For each element, add all its component to the associated list
                        lists[i].append(tensor)

                # Concatenate each component list
                return tuple(tf.concat(l, axis=0) for l in lists)

            with tf.name_scope(self.name):
                    id_tensor = tf.squeeze(tf.convert_to_tensor(file_list, dtype=tf.string))
                    # Create dataset from list of file names
                    file_list_ds = tf.data.Dataset.from_tensor_slices(id_tensor)

                    if self.mode is self.MODES.TRAIN:
                        # shuffle and repeat n_poch times if in training mode
                        file_list_ds = file_list_ds.shuffle(buffer_size=self.n_files).repeat(count=n_epochs)
                        # map the dataset with read wrapper to generate sample example and labels

                    # read data from file using the _read_wrapper
                    ds = file_list_ds.map(map_func=self._read_wrapper, num_parallel_calls=read_threads)

                    ds_obj, ds_bkg = self._zip_data_elements_tensorwise(ds)

                    # shuffle, batch prefetch
                    obj_buffer_part = int(self.sample_buffer_size * (cfg.percent_of_object_samples / 100))
                    bkg_buffer_part = self.sample_buffer_size - obj_buffer_part
                    ds_obj = ds_obj.shuffle(buffer_size=obj_buffer_part, seed=self.seed)
                    ds_bkg = ds_bkg.shuffle(buffer_size=bkg_buffer_part, seed=self.seed)

                    # no smaller final batch
                    obj_batch_part = cfg.batch_size_train * cfg.percent_of_object_samples // 100
                    bkg_batch_part = cfg.batch_size_train - obj_batch_part
                    ds_obj = ds_obj.batch(batch_size=obj_batch_part, drop_remainder=True)
                    ds_bkg = ds_bkg.batch(batch_size=bkg_batch_part, drop_remainder=True)

                    zipped_ds = tf.data.Dataset.zip((ds_obj, ds_bkg))
                    ds = zipped_ds.map(concat)

                    # already prefetch the next element to reduce latency
                    ds = ds.prefetch(1)

        return ds

    def _zip_data_elements_tensorwise(self, ds):
        """!
                here each element corresponds to one file.
                Use flat map to make each element correspond to one Sample.
                Overwrite this function if not using [sample, label]

               Parameters
               ----------
               ds : tf.data.Dataset
                   Dataset

               Returns
               -------
               ds : tf.data.Dataset
                    Dataset where each element corresponds to one sample

               """

        # def split(*element):
        #     lists = [[]for _ in range(4)]
        #     for i, tensor in enumerate(element):
        #         # For each element, add all its component to the associated list
        #         lists[i].append(tensor)
        #
        #     ds_obj = tuple((lists[0], lists[1]))
        #     ds_bkg = tuple((lists[2], lists[3]))
        #
        #     print(ds_obj, ds_bkg)
        #
        #     return tuple((ds_obj, ds_bkg))

        if self.mode is self.MODES.APPLY:
            ds = ds.flat_map(lambda e: tf.data.Dataset.from_tensor_slices(e))
            return ds
        else:
            # here each element corresponds to one file. use flat map to make each element correspond to one tensor
            # ds = ds.flat_map(lambda e_1, l_1, e_2, l_2:
            #       tf.data.Dataset.zip((
            #       tf.data.Dataset.from_tensor_slices(e_1).concatenate(tf.data.Dataset.from_tensor_slices(e_2)),
            #       tf.data.Dataset.from_tensor_slices(l_1).concatenate(tf.data.Dataset.from_tensor_slices(l_2)))))
            # ds = ds.flat_map(lambda e_1, l_1, e_2, l_2:
            #       tf.data.Dataset.zip((
            #       tf.data.Dataset.from_tensor_slices(e_1),
            #       tf.data.Dataset.from_tensor_slices(l_1),
            #       tf.data.Dataset.from_tensor_slices(e_2),
            #       tf.data.Dataset.from_tensor_slices(l_2))))
            ds_obj = ds.flat_map(lambda e, l, _1, _2: tf.data.Dataset.zip((
                  tf.data.Dataset.from_tensor_slices(e),
                  tf.data.Dataset.from_tensor_slices(l))))
            ds_bkg = ds.flat_map(lambda _1, _2, e, l: tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(e),
                tf.data.Dataset.from_tensor_slices(l))))

            # ds_obj, ds_bkg = ds.map(split)
            return ds_obj, ds_bkg

    def _read_wrapper(self, id_data_set):
        # this has been adapted from https://github.com/DLTK/DLTK

        """!
         Wrapper for the _read_file() function

        Wraps the _read_file() function and handles tensor shapes and data types

        Parameters
        ----------
        id_data_set : list
            list of tf.Tensors from the id_list queue. Provides an identifier for the examples to read.


        Returns
        -------
        list
            list of tf.Tensors read for this example

        @sa LitsLoader._read_file()


        @todo Alena, ist it _read_file function or _read_sample function?? it was earlier _read_sample function
        @todo This might have been _read_sample in an older version, but read_file makes more sense for our usage
        @todo f() is not included in the documentation !!
        @todo Nadia: modify for doxygen
        """

        def get_sample_tensors_from_file_name(file_id):
            """!
            Wrapper for the python function

            Handles the data types of the py_func

            Parameters
            ----------
            id_data_set : list
            list of tf.Tensors from the id_list queue. Provides an identifier for the examples to read.

            Returns
            -------
            list
                list of things just read
            """
            try:
                samples_obj, samples_bkg = self._read_file_and_return_numpy_samples(file_id.numpy())
            except Exception as e:
                logger.error('got error `%s from `_read_file`:', format(e))
                logger.error(traceback.format_exc())
                raise
            return samples_obj[0], samples_obj[1], samples_bkg[0], samples_bkg[1]

        ex = tf.py_function(get_sample_tensors_from_file_name, [id_data_set], (*self.dtypes, *self.dtypes))  # use python function in tensorflow

        ex[0].set_shape([None] + list(self.dshapes[0]))
        ex[1].set_shape([None] + list(self.dshapes[1]))
        ex[2].set_shape([None] + list(self.dshapes[0]))
        ex[3].set_shape([None] + list(self.dshapes[1]))

        return ex

    def _select_indices(self, data, lbl):
        return self._select_ratio_indices(data, lbl, cfg.samples_per_volume, cfg.percent_of_object_samples)

    def _select_ratio_indices(self, data, lbl, samples_per_volume, percent_of_object_samples):
        '''!
        select slice indices from which samples should be taken

        @param lbl <em>numpy array,  </em> label image, where background is zero

        Z-axis / craniocaudal axis is assumed to be third dimension. Across the third dimension the slices are differentiated
        into zero (empty slices/no segmentation) and non-zero slices.
        For <tt>  self.mode </tt> = @p 'train' all non-zero slices are added to the index list.
        For <tt>  self.mode </tt> = @p 'validate' the number of slices is restricted by config.validation_num_non_zero_indices.
        The number of zero samples is calculated in relation to the non-zero samples based on config.non_zero_sample_percent. Should this number
        be higher than the number actual number of zero indices, then only the available indices are added.

        @return slice indices as numpy array
        '''

        def _distribute_samples_accross_indices(slices, number_of_samples):
            if slices.size > 0:
                sorted_slice_indices = np.argsort(slices)
                sorted_slice_indices = sorted_slice_indices[slices[sorted_slice_indices] > 0]
                if number_of_samples <= sorted_slice_indices.size:
                    selected_slices = np.random.choice(sorted_slice_indices, number_of_samples,
                                                       replace=False)  # make sure selection is unique
                    samples_per_slice = np.ones(selected_slices.size, dtype=np.int)
                else:
                    # take at least one sample from each slice and then determin the rest
                    selected_slices = sorted_slice_indices
                    samples_per_slice = np.ones(selected_slices.size, dtype=np.int)

                    s_r = int(max(np.floor(number_of_samples / sorted_slice_indices.size), 1))
                    samples_per_slice = samples_per_slice * s_r
                    number_of_missing_samples = int(number_of_samples - np.sum(samples_per_slice))

                    # slices at the end of the list have the most center voxels
                    if number_of_missing_samples > 0:
                        samples_per_slice[-number_of_missing_samples:] = s_r + 1

                    assert np.sum(samples_per_slice) == number_of_samples
            else:
                selected_slices = np.array([])  # if there are no non object slices, pass empty array
                samples_per_slice = np.array([])

            # print('  Selected Slices: ', selected_slices.size, 'Samples per Slice:', samples_per_slice.size)

            return selected_slices, samples_per_slice

        data_shape = data.shape
        if self.data_rank == 3:
            bb_dim = [self.dshapes[0][0], self.dshapes[0][1]]
        else:
            bb_dim = [self.dshapes[0][1], self.dshapes[0][2]]
        min_x = bb_dim[0] // 2
        max_x = data_shape[0] - bb_dim[0] // 2
        min_y = bb_dim[1] // 2
        max_y = data_shape[1] - bb_dim[1] // 2
        s = lbl.shape[2]  # third dimension is z-axis
        indices = np.arange(0 + self.slice_shift, s - self.slice_shift, 1)
        valid_patch_centers_objects = (lbl > 0)[min_x:max_x, min_y:max_y, indices]
        valid_patch_centers_background = np.logical_not(valid_patch_centers_objects)
        # print('---- Valid Patch Centers:', valid_patch_centers_objects.size, ' - OB:',
        #       np.sum(valid_patch_centers_objects)/valid_patch_centers_objects.size,
        #       ' - BG:', np.sum(valid_patch_centers_background)/valid_patch_centers_objects.size)
        n_object_per_slice = np.sum(valid_patch_centers_objects, axis=(0, 1))
        n_background_per_slice = np.sum(valid_patch_centers_background, axis=(0, 1))
        if np.sum(n_object_per_slice) == 0:
            raise Exception('All labels are zero, no objects were found, either the labels are incorrect or there was a problem processing the image.')
        # print('---- Centers per Slice:', valid_patch_centers.size, ' - OB:', n_object_per_slice. size, n_object_per_slice, ' - BG:', n_background_per_slice.size, n_background_per_slice)
        # If the segmentation is not a continuous volume (case 102), a vector from min to max leads to empty slices.
        # -> use unique instead!
        n_object_samples = (samples_per_volume * percent_of_object_samples) // 100
        n_background_samples = samples_per_volume - n_object_samples
        # print('Sample Ratio: ', n_object_samples, n_background_samples)
        # print(' ---- Objects ---- ')
        object_indices, samples_per_slice_obj = _distribute_samples_accross_indices(n_object_per_slice, n_object_samples)
        object_indices += self.slice_shift
        # print(' ---- Background ---- ')
        background_indices, samples_per_slice_bkg = _distribute_samples_accross_indices(n_background_per_slice, n_background_samples)
        background_indices += self.slice_shift

        # print('          Slices:', s, self.slice_shift, 'Number of Indices (Object, Background): ', object_indices.size, background_indices.size)
        # print('            Indices (Object, Background): ', object_indices, background_indices)
        # print('            Samples per Slice (Object, Background): ', samples_per_slice_obj, samples_per_slice_bkg)
        # print('---- Object Index List:', np.min(object_indices), np.max(object_indices))
        # print('---- Background Index List:', np.min(background_indices), np.max(background_indices))
        return object_indices, samples_per_slice_obj, background_indices, samples_per_slice_bkg

    def _get_samples_from_volume(self, data, lbl):
        return self._get_ratio_samples_from_volume(data, lbl)

    @tf.autograph.experimental.do_not_convert
    def _get_ratio_samples_from_volume(self, data, lbl):
        '''!
        Get batches/samples for training from image and label data.

        @param data <em>numpy array,  </em> input image data with config.num_channels channels
        @param lbl <em>numpy array,  </em> label image, one channel

        This function operates as follows:
        - calls _select_indices()
        - calls _get_samples() for each index
        - calls _preprocess() on the input samples

        @return numpy arrays I (containing input samples) and L (containing according labels)
        '''
        indices_obj, sampling_rates_obj, indices_bkg, sampling_rates_bkg = self._select_indices(data, lbl)

        L_obj = np.zeros((np.sum(sampling_rates_obj), *self.dshapes[1]))
        I_obj = np.zeros((np.sum(sampling_rates_obj), *self.dshapes[0]))
        # print(L_obj.shape)
        current_index = 0
        for i in range(len(indices_obj)):
            # print('   Object #', i, indices_obj[i], sampling_rates_obj[i], ':', current_index)
            I_obj[current_index:current_index+sampling_rates_obj[i]], L_obj[current_index:current_index+sampling_rates_obj[i]]\
                = self._get_samples_by_index(data, lbl, indices_obj[i], sampling_rates_obj[i], object_sampling=True)
            current_index = np.sum(sampling_rates_obj[:i+1])

        L_bkg = np.zeros((np.sum(sampling_rates_bkg), *self.dshapes[1]))
        I_bkg = np.zeros((np.sum(sampling_rates_bkg), *self.dshapes[0]))
        # print(L_bkg.shape)
        current_index = 0
        for i in range(len(indices_bkg)):
            # print('   Background #', i, indices_bkg[i], sampling_rates_bkg[i], current_index)
            I_bkg[current_index:current_index + sampling_rates_bkg[i]], L_bkg[current_index:current_index +
                                                                                            sampling_rates_bkg[i]] \
                = self._get_samples_by_index(data, lbl, indices_bkg[i], sampling_rates_bkg[i], object_sampling=False)
            current_index = np.sum(sampling_rates_bkg[:i+1])

        if self.mode is self.MODES.TRAIN:
            I_bkg, L_bkg = self._augment_samples(I_bkg, L_bkg)
            I_obj, L_obj = self._augment_samples(I_obj, L_obj)

        logger.debug('          Image Samples Shape: %s (obj) %s (bkg)', I_obj.shape, I_bkg.shape)
        logger.debug('          Label Samples Shape: %s (obj) %s (bkg)', L_obj.shape, L_bkg.shape)

        return [I_obj, L_obj], [I_bkg, L_bkg]
