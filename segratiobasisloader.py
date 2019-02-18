import tensorflow as tf
import traceback
import warnings
import numpy as np
from . import config as cfg
from .segbasisloader import SegBasisLoader


class SegRatioBasisLoader(SegBasisLoader):

    def __call__(self, file_list, batch_size, n_epochs, read_threads):
        """!
        Parses the list of file IDs given in @p file_list, identifies .nii files to load and loads the data and labels, genetares @p tf.dataset of data and labels with necessary repeating, shuffling, batching and prefetching.

        @param  file_list : <em>  array of strings,  </em> where each string is a file ID corresponding to a pair of data file and label file to be loaded. @p file_list should be obtained from a @p .csv file and then converted to numpy array. Each ID string should have the format <tt> 'Location\\file_number'</tt>. From @p Location, the data file and label file with the @p file_number, respectiely named as @p volume-file_number.nii and segmentation-file_number.nii are loaded. (See also LitsLoader._read_file(), LitsLoader._load_file() for more details.)
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

        def concat(*ds_elements):
            #Create one empty list for each component of the dataset
            lists = [[] for _ in ds_elements[0]]
            for element in ds_elements:
                for i, tensor in enumerate(element):
                    #For each element, add all its component to the associated list
                    lists[i].append(tensor)

            #Concatenate each component list
            return tuple(tf.concat(l, axis=0) for l in lists)

        with tf.name_scope(self.name):
            id_tensor = tf.squeeze(tf.convert_to_tensor(file_list, dtype=tf.string))
            # Create dataset from list of file names,
            file_list_ds = tf.data.Dataset.from_tensor_slices(id_tensor)
            #shuffle and repeat n_poch times if in training mode
            file_list_ds = file_list_ds.shuffle(buffer_size=self.file_name_buffer_size).repeat(count=n_epochs)
            # map the dataset with read wrapper to generate sample example and labels
            ds = file_list_ds.map(map_func=self._read_wrapper, num_parallel_calls=read_threads)

            # here each element corresponds to one file. use flat map to make each element correspond to one tensor
            ds_obj = ds.flat_map(lambda e, l, _1, _2: tf.data.Dataset().zip((
                  tf.data.Dataset().from_tensor_slices(e),
                  tf.data.Dataset().from_tensor_slices(l))))
            ds_bkg = ds.flat_map(lambda _1, _2, e, l: tf.data.Dataset().zip((
                tf.data.Dataset().from_tensor_slices(e),
                tf.data.Dataset().from_tensor_slices(l))))

            # shuffle, batch prefetch
            obj_buffer_part = int(self.train_buffer_size * (cfg.percent_of_object_samples / 100))
            bkg_buffer_part = self.train_buffer_size - obj_buffer_part
            ds_obj = ds_obj.shuffle(buffer_size=obj_buffer_part, seed=self.seed)
            ds_bkg = ds_bkg.shuffle(buffer_size=bkg_buffer_part, seed=self.seed)

            # no smaller final batch
            obj_batch_part = cfg.batch_size * cfg.percent_of_object_samples // 100
            bkg_batch_part = cfg.batch_size - obj_batch_part
            ds_obj = ds_obj.batch(batch_size=obj_batch_part, drop_remainder=True)
            ds_bkg = ds_bkg.batch(batch_size=bkg_batch_part, drop_remainder=True)

            zipped_ds = tf.data.Dataset.zip((ds_obj, ds_bkg))
            ds = zipped_ds.map(concat)
            ds = ds.prefetch(1)
        return ds

    def _read_wrapper(self, id_queue, **kwargs):
        # this has been adapted from https://github.com/DLTK/DLTK

        """!
         Wrapper for the _read_file() function

        Wraps the _read_file() function and handles tensor shapes and data types

        Parameters
        ----------
        id_queue : list
            list of tf.Tensors from the id_list queue. Provides an identifier for the examples to read.
        kwargs :
            additional arguments for the '_read_sample function'

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

        def get_tensors_from_file_names(id):
            """!
            Wrapper for the python function

            Handles the data types of the py_func

            Parameters
            ----------
            id_queue : list
            list of tf.Tensors from the id_list queue. Provides an identifier for the examples to read.

            Returns
            -------
            list
                list of things just read
            """
            try:
                ex_obj, ex_bkg = self._read_file(id)
            except Exception as e:
                print('got error `{} from `_read_file`:'.format(e))
                print(traceback.format_exc())
                raise

            # eventually fix data types of read objects
            tensors_obj = []
            for t, d in zip(ex_obj, self.dtypes):
                if isinstance(t, np.ndarray):
                    tensors_obj.append(t.astype(self._map_dtype(d)))
                elif isinstance(t, (float, int)):
                    if d is tf.float32 and isinstance(t, int):
                        warnings.warn('Losing accuracy by converting int to float')
                        tensors_obj.append(self._map_dtype(d)(t))
                elif isinstance(t, bool):
                    tensors_obj.append(t)
                else:
                    raise Exception('Not sure how to interpret "{}"'.format(type(t)))

            tensors_bkg = []
            for t, d in zip(ex_bkg, self.dtypes):
                if isinstance(t, np.ndarray):
                    tensors_bkg.append(t.astype(self._map_dtype(d)))
                elif isinstance(t, (float, int)):
                    if d is tf.float32 and isinstance(t, int):
                        warnings.warn('Losing accuracy by converting int to float')
                        tensors_bkg.append(self._map_dtype(d)(t))
                elif isinstance(t, bool):
                    tensors_bkg.append(t)
                else:
                    raise Exception('Not sure how to interpret "{}"'.format(type(t)))

            return tensors_obj[0], tensors_obj[1], tensors_bkg[0], tensors_bkg[1]

        ex = tf.py_func(get_tensors_from_file_names, [id_queue], (*self.dtypes, *self.dtypes))  # use python function in tensorflow

        ex[0].set_shape([None] + list(self.dshapes[0]))
        ex[1].set_shape([None] + list(self.dshapes[1]))
        ex[2].set_shape([None] + list(self.dshapes[0]))
        ex[3].set_shape([None] + list(self.dshapes[1]))

        return ex

    def _select_indices(self, data, lbl):
        raise NotImplementedError('not implemented')

    def _select_ratio_indices(self, data, lbl, samples_per_object_slice):
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
        s = lbl.shape[2]  # third dimension is z-axis
        indices = np.arange(0 + cfg.slice_shift, s - cfg.slice_shift, 1)
        object_indices = np.intersect1d(np.unique(np.nonzero(lbl)[2]), indices)  # all valid slices containing labels
        z_bkg = np.setdiff1d(indices, object_indices)  # select all slices which do not contain annotations
        # If the segmentation is not a continuous volume (case 102), a vector from min to max leads to empty slices.
        # -> use unique instead!

        if z_bkg.size > 0:  # some scans don't have non object slices
            # select 100 - cfg.non_zero_sample_percent empty samples, but pay attention to samples per slice
            number_of_background_indices = min(int(((object_indices.size * samples_per_object_slice)
                                              / cfg.percent_of_object_samples) * (100 - cfg.percent_of_object_samples)
                                             / cfg.samples_per_slice_bkg), z_bkg.size)
            selection = np.random.choice(z_bkg.size, number_of_background_indices,
                                         replace=False)  # make sure selection is unique
            background_indices = z_bkg[selection]
        else:
            background_indices = np.array([])  # if there are no non object slices, pass empty array

        print('      Slices:', s, 'Number of Indices (Object, Background): ', object_indices.size, background_indices.size)

        return object_indices, background_indices

    def _get_samples_from_volume(self, data, lbl):
        raise NotImplementedError('not implemented')

    def _get_ratio_samples_from_volume(self, data, lbl, n_object_samples, n_background_samples):
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
        indices_obj, indices_bkg = self._select_indices(data, lbl)

        L_obj = np.zeros((len(indices_obj)*n_object_samples, *self.dshapes[1]))
        I_obj = np.zeros((len(indices_obj)*n_object_samples, *self.dshapes[0]))
        for i in range(len(indices_obj)):
            images, labels = self._get_samples_by_index(data, lbl, indices_obj[i], n_object_samples)
            I_obj[i*n_object_samples:(i+1)*n_object_samples] = images
            L_obj[i*n_object_samples:(i+1)*n_object_samples] = labels

        L_bkg = np.zeros((len(indices_bkg) * cfg.samples_per_slice_bkg, *self.dshapes[1]))
        I_bkg = np.zeros((len(indices_bkg) * cfg.samples_per_slice_bkg, *self.dshapes[0]))
        for i in range(len(indices_bkg)):
            images, labels = self._get_samples_by_index(data, lbl, indices_bkg[i], n_background_samples)
            I_bkg[i * cfg.samples_per_slice_bkg:(i + 1) * cfg.samples_per_slice_bkg] = images
            L_bkg[i * cfg.samples_per_slice_bkg:(i + 1) * cfg.samples_per_slice_bkg] = labels

        I_bkg = self._preprocess(I_bkg)
        I_obj = self._preprocess(I_obj)

        print('    Image Shape: ', I_obj.shape, I_bkg.shape)
        print('    Label Shape: ', L_obj.shape, L_bkg.shape)

        return [I_obj, L_obj], [I_bkg, L_bkg]







    # def _get_batches(self, data, lbl):
    #     '''!
    #     Get batches/samples for training from image and label data.
    #
    #     @param data <em>numpy array,  </em> input image data with config.num_channels channels
    #     @param lbl <em>numpy array,  </em> label image, one channel
    #
    #     This function operates as follows:
    #     - calls _select_indices()
    #     - calls _get_samples() for each index
    #     - calls _preprocess() on the input samples
    #
    #     @return numpy arrays I (containing input samples) and L (containing according labels)
    #     '''
    #     lesion_indices, liver_indices, indices_bkg = self._select_indices(lbl)
    #
    #     L_lesion = np.zeros((len(lesion_indices)*cfg.samples_per_slice_lesion, *self.dshapes[1]))
    #     I_lesion = np.zeros((len(lesion_indices)*cfg.samples_per_slice_lesion, *self.dshapes[0]))
    #     for i in range(len(lesion_indices)):
    #         images, labels = self._get_samples(data, lbl, lesion_indices[i], cfg.samples_per_slice_lesion)
    #         I_lesion[i*cfg.samples_per_slice_lesion:(i+1)*cfg.samples_per_slice_lesion] = images
    #         L_lesion[i*cfg.samples_per_slice_lesion:(i+1)*cfg.samples_per_slice_lesion] = labels
    #
    #     L_liver = np.zeros((len(liver_indices) * cfg.samples_per_slice_liver, *self.dshapes[1]))
    #     I_liver = np.zeros((len(liver_indices) * cfg.samples_per_slice_liver, *self.dshapes[0]))
    #     for i in range(len(liver_indices)):
    #         images, labels = self._get_samples(data, lbl, liver_indices[i], cfg.samples_per_slice_liver)
    #         I_liver[i * cfg.samples_per_slice_liver:(i + 1) * cfg.samples_per_slice_liver] = images
    #         L_liver[i * cfg.samples_per_slice_liver:(i + 1) * cfg.samples_per_slice_liver] = labels
    #
    #     L_obj = np.concatenate([L_lesion, L_liver], 0)
    #     I_obj = np.concatenate([I_lesion, I_liver], 0)
    #
    #     L_bkg = np.zeros((len(indices_bkg) * cfg.samples_per_slice_bkg, *self.dshapes[1]))
    #     I_bkg = np.zeros((len(indices_bkg) * cfg.samples_per_slice_bkg, *self.dshapes[0]))
    #     for i in range(len(indices_bkg)):
    #         images, labels = self._get_samples(data, lbl, indices_bkg[i], cfg.samples_per_slice_bkg)
    #         I_bkg[i * cfg.samples_per_slice_bkg:(i + 1) * cfg.samples_per_slice_bkg] = images
    #         L_bkg[i * cfg.samples_per_slice_bkg:(i + 1) * cfg.samples_per_slice_bkg] = labels
    #
    #     I_bkg = self._preprocess(I_bkg)
    #     I_obj = self._preprocess(I_obj)
    #
    #     print('   Image Shape: ', I_obj.shape, I_bkg.shape)
    #     print('   Label Shape: ', L_obj.shape, L_bkg.shape)
    #
    #     return [I_obj, L_obj], [I_bkg, L_bkg]
    #
    # def _select_indices(self, lbl):
    #     '''!
    #     select slice indices from which samples should be taken
    #
    #     @param lbl <em>numpy array,  </em> label image, where background is zero
    #
    #     Z-axis / craniocaudal axis is assumed to be third dimension. Across the third dimension the slices are differentiated
    #     into zero (empty slices/no segmentation) and non-zero slices.
    #     For <tt>  self.mode </tt> = @p 'train' all non-zero slices are added to the index list.
    #     For <tt>  self.mode </tt> = @p 'validate' the number of slices is restricted by config.validation_num_non_zero_indices.
    #     The number of zero samples is calculated in relation to the non-zero samples based on config.non_zero_sample_percent. Should this number
    #     be higher than the number actual number of zero indices, then only the available indices are added.
    #
    #     @return slice indices as numpy array
    #     '''
    #     s = lbl.shape[2]  # third dimension is z-axis
    #     indices = np.arange(0 + cfg.slice_shift, s - cfg.slice_shift, 1)
    #
    #     nz_liver = np.intersect1d(np.unique(np.nonzero(lbl)[2]), indices)  # all valid slices containing liver
    #     z_bkg = np.setdiff1d(indices, nz_liver)  # select all slices which do not contain annotations
    #     nz_lesion = np.nonzero(np.greater(lbl, 1))
    #     # If the segmentation is not a continuous volume (case 102), a vector from min to max leads to empty slices.
    #     # -> use unique instead!
    #     lesion_indices = np.random.permutation(np.intersect1d(np.unique(nz_lesion[2]), indices))
    #     nz_onlyliver = np.setdiff1d(nz_liver, nz_lesion)
    #
    #     # if more lesion samples than threshold use 50:50 lesion and only liver slices
    #     if lesion_indices.size > cfg.min_n_samples:
    #         # check if there are enough only liver slices
    #         if nz_onlyliver.size > 0:
    #             n_liversamples = min(lesion_indices.size, nz_onlyliver.size)  # 50:50 or all there are
    #             selection = np.random.choice(nz_onlyliver.size, n_liversamples, replace=False) # make sure selection is unique
    #             #print('Sizes: ', s, lesion_indices.size, nz_onlyliver.size, 'Min: ', n_liversamples, type(n_liversamples), selection.size, type(selection))
    #             liver_indices = nz_onlyliver[selection]
    #         else:
    #             liver_indices = np.array([])  # if there are no only liver slices, pass empty array
    #     else:
    #         #print('  Few Lesion samples available: ', lesion_indices.size, '. Using ', cfg.min_n_samples, 'Liver Samples instead.')
    #         selection = np.random.choice(nz_onlyliver.size, min(cfg.min_n_samples, nz_onlyliver.size), replace=False)
    #         liver_indices = nz_onlyliver[selection]
    #
    #     if z_bkg.size > 0:  # some scans don't have non liver slices
    #         # select 100 - cfg.non_zero_sample_percent empty samples, but pay attention to samples per slice
    #         number_of_zero_indices = min(int(((liver_indices.size * cfg.samples_per_slice_liver
    #                                             + lesion_indices.size * cfg.samples_per_slice_lesion)
    #                                           / cfg.percent_of_object_samples) * (100 - cfg.percent_of_object_samples)
    #                                          / cfg.samples_per_slice_bkg), z_bkg.size)
    #         selection = np.random.choice(z_bkg.size, number_of_zero_indices,
    #                                      replace=False)  # make sure selection is unique
    #         zero_indices = z_bkg[selection]
    #     else:
    #         zero_indices = np.array([])  # if there are no non liver slices, pass empty array
    #
    #     print('    Slices:', s, 'Number of Indices (Lesion, Liver, Zero): ', lesion_indices.size, liver_indices.size, zero_indices.size)
    #
    #     return lesion_indices, liver_indices, zero_indices
    #
    # def one_hot_label(label):
    #     '''!
    #     convert a one-channel binary label image into a one-hot label image
    #
    #     @param label <em>numpy array,  </em> label image, where background is zero and object is 1 (only works for binary problems)
    #
    #     @return two-channel one-hot label as numpy array
    #     '''
    #     # add empty last dimension
    #     if label.ndim < 4 and label.shape[-1] > 1:
    #         label = np.expand_dims(label, axis=-1)
    #     # add empty first dimension if single sample
    #     if label.ndim < 3:
    #         label = np.expand_dims(label, axis=0)
    #
    #     liver_label = np.equal(label, 1)
    #     lesion_label = np.equal(label, 2)
    #
    #     invert_label = np.logical_not(np.logical_and(liver_label, lesion_label))  # complementary binary mask
    #     return np.concatenate([invert_label, liver_label, lesion_label], axis=-1)  # fuse
