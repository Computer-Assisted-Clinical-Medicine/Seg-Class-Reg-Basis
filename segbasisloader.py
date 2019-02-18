import numpy as np
import SimpleITK as sitk
from . import config as cfg
from .NetworkBasis import image as Image
from .NetworkBasis.dataloader import DataLoader


class SegBasisLoader(DataLoader):
    """!
    LitsLoader Class
    """

    def _set_up_shapes_and_types(self):
        self.n_channels = cfg.num_channels
        self.dtypes = [cfg.dtype, cfg.dtype]
        self.dshapes = [cfg.train_input_shape, cfg.train_label_shape]
        self.slice_shift = ((self.n_channels - 1) // 2) * cfg.in_between_slice_factor

    def _set_up_capacities(self):
        if self.mode == 'train':
            self.file_name_buffer_size = cfg.file_name_capacity
            self.train_buffer_size = cfg.batch_capacity
        elif self.mode == 'validate':
            self.file_name_buffer_size = cfg.file_name_capacity // 10
            self.train_buffer_size = cfg.batch_capacity_valid

    def _read_file(self, file_id):
        """!
        reads nii data file and nii labels file given the @p file_id
        @param file_id <em>string,  </em> a file ID correspondinging to a pair of data file and label file to be loaded. It should have the format <tt> 'Location\\file_number'</tt> and identify a pair of data file and labels file. @p Location must contain the data file and label file with that @p file_number, respectiely named as @p volume-file_number.nii and segmentation-file_number.nii. For example, <tt>'C:\\DataLoction\\0'</tt> can be a @p file_id. Then <tt> C:\\DataLoction </tt> should contain @p volume-0.nii and @p segmetation-0.nii.


        Given the @p file_id, this function
        - loads data and labels (See _load_file() )
        - todo _get_batches()

        @todo Nadia: add more detail how it works, write about _get_batches
        """
        data, lbl = self._load_file(file_id)
        ex_obj, ex_bkg = self._get_samples_from_volume(data, lbl)
        #print('    ', file_id, 'Object: ', np.max((ex_obj[0])), np.max((ex_obj[1])), 'Background: ', np.max((ex_bkg[0])), np.max((ex_bkg[1])))
        return ex_obj, ex_bkg

    def _load_file(self, path):
        raise NotImplementedError('not implemented')

    def _get_samples_from_volume(self, data, lbl):
        '''!
        Get batches/samples for training from image and label data.

        @param data <em>numpy array,  </em> input image data with config.num_channels channels
        @param lbl <em>numpy array,  </em> label image, one channel

        This function operates as follows:
        - calls _select_indices()
        - calls _get_samples_by_index() for each index
        - calls _preprocess() on the input samples

        @return numpy arrays I (containing input samples) and L (containing according labels)
        '''
        indices = self._select_indices(data, lbl)
        L = np.zeros((len(indices)*cfg.samples_per_slice_uni, *self.dshapes[1]))
        I = np.zeros((len(indices)*cfg.samples_per_slice_uni, *self.dshapes[0]))

        for i in range(len(indices)):
            images, labels = self._get_samples_by_index(data, lbl, indices[i])
            I[i*cfg.samples_per_slice_uni:(i+1)*cfg.samples_per_slice_uni] = images
            L[i*cfg.samples_per_slice_uni:(i+1)*cfg.samples_per_slice_uni] = labels

        I = self._preprocess(I)

        print('   Image Shape: ', I.shape)
        print('   Label Shape: ', L.shape)

        return [I, L]

    def _select_indices(self, data, lbl):
        raise NotImplementedError('not implemented')

    def _get_samples_by_index(self, data, label, index, samples_per_slice=cfg.samples_per_slice_uni):
        '''!
        Get samples from index slice.

        @param data <em>numpy array,  </em> input image
        @param label <em>numpy array,  </em> label image
        @param index <em>int,  </em> index of the slices that should be processed

        This is for Training.
        This function operates as follows:
        - Extract current image and label for index.
        - calls __select_patches() on the current data
        - calls one_hot_label() on the label samples

        @return numpy arrays I (containing input samples) and L (containing according labels)
        '''
        image = data[:, :, index - cfg.slice_shift:index + cfg.slice_shift + 1:cfg.in_between_slice_factor]
        lbl = label[:, :, index]

        I, lbl = self._select_patches(image, lbl, samples_per_slice)
        L = self.__class__.one_hot_label(lbl)

        return [I, L]

    def _select_patches(self, image, label, samples_per_slice):
        '''!
        extracts sample patches from the given image slice

        @param image <em>numpy array,  </em> input image data with config.num_channels channels
        @param label <em>numpy array,  </em> label image, one channel

        Square patches are extracted from the data. Patch centers are restricted to positions so that the whole
        patch is covered by the image. If the current slice contains the object in the label image, the patch center is
        further restricted to mean +/- 3 * std of the object position in each direction.
        The number of patches taken from each slice is defined by config.samples_per_slice.

        @return numpy arrays I (containing input patches) and L (containing label patches)
        '''
        n_z = np.nonzero(label)
        data_shape = image.shape
        bb_dim = [self.dshapes[0][0], self.dshapes[0][1]]

        if n_z[0].size == 0 and not cfg.random_sampling_mode == cfg.SAMPLINGMODES.UNIFORM:
            # if there are no lables on the slice, sample in the body
            n_z = np.nonzero(np.greater(image, cfg.norm_min_v))

        if n_z[0].size > 0 and cfg.random_sampling_mode == cfg.SAMPLINGMODES.CONSTRAINED_MUSTD:
            mean_x = np.mean(n_z[0], dtype=np.int32)
            std_x = np.std(n_z[0], dtype=np.int32) * cfg.patch_shift_factor
            mean_y = np.mean(n_z[1], dtype=np.int32)
            std_y = np.std(n_z[1], dtype=np.int32) * cfg.patch_shift_factor

            # ensure that bounding box is inside image
            min_x = min(max(mean_x - std_x, bb_dim[0] // 2), data_shape[0] - bb_dim[0] // 2)
            max_x = max(min(mean_x + std_x + 1, data_shape[0] - bb_dim[0] // 2), bb_dim[0] // 2)
            min_y = min(max(mean_y - std_y, bb_dim[1] // 2), data_shape[1] - bb_dim[1] // 2)
            max_y = max(min(mean_y + std_y, data_shape[1] - bb_dim[1] // 2), bb_dim[1] // 2)
        else:
            min_x = bb_dim[0] // 2
            max_x = data_shape[0] - bb_dim[0] // 2
            min_y = bb_dim[1] // 2
            max_y = data_shape[1] - bb_dim[1] // 2

        L = np.zeros((samples_per_slice, *cfg.train_label_shape[:-1]))
        I = np.zeros((samples_per_slice, *cfg.train_input_shape))

        # select samples
        if cfg.random_sampling_mode == cfg.SAMPLINGMODES.UNIFORM or cfg.random_sampling_mode == cfg.SAMPLINGMODES.CONSTRAINED_MUSTD:
            sample_x = np.random.randint(min_x, max_x + 1, samples_per_slice)
            sample_y = np.random.randint(min_y, max_y + 1, samples_per_slice)
            for i in range(samples_per_slice):
                I[i] = image[sample_x[i] - (bb_dim[0] // 2):sample_x[i] + (bb_dim[0] // 2),
                         sample_y[i] - (bb_dim[1] // 2):sample_y[i] + (bb_dim[1] // 2), :]
                L[i] = label[sample_x[i] - (bb_dim[0] // 2):sample_x[i] + (bb_dim[0] // 2),
                         sample_y[i] - (bb_dim[1] // 2):sample_y[i] + (bb_dim[1] // 2)]
        elif cfg.random_sampling_mode == cfg.SAMPLINGMODES.CONSTRAINED_LABEL:
            # TODO: select patch centers from label positions
            pass
        else:
            #TODO: throw error for not allowed mode
            pass

        return [I, L]

    @staticmethod
    def one_hot_label(label):
        '''!
        convert a one-channel binary label image into a one-hot label image

        @param label <em>numpy array,  </em> label image, where background is zero and object is 1 (only works for binary problems)

        @return two-channel one-hot label as numpy array
        '''
        # add empty last dimension
        if label.ndim < 4 and label.shape[-1] > 1:
            label = np.expand_dims(label, axis=-1)
        # add empty first dimension if single sample
        if label.ndim < 3:
            label = np.expand_dims(label, axis=0)

        invert_label = np.logical_not(label)  # complementary binary mask
        return np.concatenate([invert_label, label], axis=-1)  # fuse

    def _preprocess(self, data):
        '''!
        preprocess

        @todo Nadia
        '''
        data = self.normalize(data)
        if data.ndim < 4:
            data = np.expand_dims(data, axis=-1)
        return data

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
        if self.mode == 'train':
            do_augment = True
        else:
            do_augment = False
        return self.resample(data, data_info, adapt_resolution=cfg.adapt_resolution, label=label, do_augment=do_augment)

    ### Static Functions ###

    @staticmethod
    def normalize(img):
        '''!
        Truncates input to interval [config.norm_min_v, config.norm_max_v] and normalizes it to interval [-1, 1].

        @todo Nadia
        '''
        flags = img < cfg.norm_min_v
        img[flags] = cfg.norm_min_v
        flags = img > cfg.norm_max_v
        img[flags] = cfg.norm_max_v
        img = (img - cfg.norm_min_v) / (cfg.norm_max_v - cfg.norm_min_v + cfg.norm_eps)
        img = (img * 2) - 1
        return img


    @staticmethod
    def resample(data, data_info, adapt_resolution=True, label=None, do_augment=False):
        '''!
        resamples <tt>data</tt> and <tt>label</tt> image using simple Simple ITK

        @param data <em>ITK image,  </em> input data
        @param data_info, current meta information of data
        @param adapt_resolution <em>bool,  </em> only if True resolution is changed
        @param label <em>ITK image,  </em> same size as <tt>data</tt>
        @param do_augment <em>bool,  </em> enables data augmentation for training

        All initial image information is taken from the <tt>data</tt> image.
        The target spacing and target direction are set in config.py.
        If <tt>do_augment</tt> is true, a random rotation around the craniocaudal axis is added.
        The extend of the rotation is restricted by cgf.max_rotation.

        @return resampled <tt>data</tt> and <tt>label</tt> as ITK images
        '''

        # how much world space is covered by the image
        x_extend = data_info['orig_spacing'][0] * data_info['orig_size'][0]
        y_extend = data_info['orig_spacing'][1] * data_info['orig_size'][1]
        z_extend = data_info['orig_spacing'][2] * data_info['orig_size'][2]
        #print('  In Extend: ', x_extend, y_extend, z_extend)

        if adapt_resolution:
            # size of the output image, so input space is covered with new resolution
            out_size = (cfg.target_size[0], cfg.target_size[1], np.int(np.round(z_extend / cfg.target_spacing[2])))
            #print('  Out Size: ', out_size)

            # When resampling, the origin has to be changed,
            # otherwise the patient will not be in the image center afterwards
            out_x_extend = cfg.target_spacing[0] * out_size[0]
            out_y_extend = cfg.target_spacing[1] * out_size[1]
            out_z_extend = cfg.target_spacing[2] * out_size[2]
            #print('  Out Extend: ', out_x_extend, out_y_extend, out_z_extend)

            # shift by half the difference in extend
            x_diff = (x_extend - out_x_extend) / 2
            y_diff = (y_extend - out_y_extend) / 2
            z_diff = (z_extend - out_z_extend) / 2

            out_spacing = cfg.target_spacing

        else:
            out_size = data_info['orig_size']
            out_spacing = data_info['orig_spacing']

        target_direction = cfg.target_direction
        # fix the direction
        #print('   Directions: ', cfg.target_direction, data_info['orig_direction'])
        if cfg.target_direction != data_info['orig_direction']:
            if cfg.target_direction[0] > data_info['orig_direction'][0]:
                data_origin_x = data_info['orig_origin'][0] - x_extend
            else:
                data_origin_x = data_info['orig_origin'][0]
            if cfg.target_direction[4] > data_info['orig_direction'][4]:
                data_origin_y = data_info['orig_origin'][1] - y_extend
            else:
                data_origin_y = data_info['orig_origin'][1]

            target_origin = (data_origin_x, data_origin_y, data_info['orig_origin'][2])
        else:
            target_origin = data_info['orig_origin']

        #print('   Origin: ', target_origin, data_info['orig_origin'], target_direction)

        if adapt_resolution:
            # Multiply with the direction to shift according to the system axes.
            out_origin = (target_origin[0] + (x_diff * target_direction[0]), target_origin[1] + (y_diff * target_direction[4]), target_origin[2] + (z_diff * target_direction[8]))
            #print('  New Origin: ', out_origin)
        else:
            out_origin = target_origin

        # if augmentation is on, do random translation and rotation
        if do_augment:
            transform = sitk.Euler3DTransform()
            # rotation center is center of the image center in world coordinates
            rotation_center = (data_info['orig_origin'][0] + (data_info['orig_direction'][0] * x_extend / 2),
                               data_info['orig_origin'][1] + (data_info['orig_direction'][4] * y_extend / 2),
                               data_info['orig_origin'][2] + (data_info['orig_direction'][8] * z_extend / 2))
            #print('Rot Center: ', rotation_center)
            transform.SetCenter(rotation_center)
            # apply a random rotation around the z-axis
            rotation = np.random.uniform(np.pi * -cfg.max_rotation, np.pi * cfg.max_rotation)
            transform.SetRotation(0, 0, rotation)
            #print('  Rotation: ', rotation)
            # apply random translation in x- and y-directions
            transform.SetTranslation((0, 0, 0))
            #print('  Translation: ', transform.GetTranslation())
        else:
            transform = sitk.Transform(3, sitk.sitkIdentity)

        # data: linear resampling, fill outside with air (-1000)
        new_data = sitk.Resample(data, out_size, transform, sitk.sitkLinear, out_origin, out_spacing,
                                 target_direction, cfg.data_background_value, cfg.target_type_image)

        if label is not None:
            # label: nearest neighbor resampling, fill with background(0)
            new_label = sitk.Resample(label, out_size, transform, sitk.sitkNearestNeighbor, out_origin, out_spacing,
                                      target_direction, cfg.label_background_value, cfg.target_type_label)
            return new_data, new_label
        else:
            return new_data