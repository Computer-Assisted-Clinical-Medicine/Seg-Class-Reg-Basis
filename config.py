"""!
@file config.py
Sets the parameters for configuration
"""
import socket
from enum import Enum

import numpy as np
import SimpleITK as sitk
import tensorflow as tf

##### Enums #####


class SAMPLINGMODES(Enum):
    UNIFORM = 0
    CONSTRAINED_MUSTD = 1
    CONSTRAINED_LABEL = 2


class NORMALIZING(Enum):
    WINDOW = 0
    MEAN_STD = 1
    PERCENT5 = 2


class NOISETYP(Enum):
    GAUSSIAN =0
    POISSON = 1

#these files are used to store the different sets
train_csv = 'train.csv'
fine_csv = 'fine.csv'
vald_csv = 'vald.csv'
test_csv = 'test.csv'

#prefixes are used for file names
sample_file_name_prefix = 'sample-' 
label_file_name_prefix = 'label-'
#the suffix determines the format
file_suffix = '.nrrd'

if socket.gethostname() == 'ckm4cad':
    ONSERVER = True
    op_parallelism_threads = 6
    batch_size_train = 16
    batch_capacity_train = 4000
    train_reader_instances = 2

else:
    ONSERVER = False
    op_parallelism_threads = 3
    batch_size_train = 4
    batch_capacity_train = 400
    train_reader_instances = 1


summary_steps_per_epoch = 2 # how often a summary is calculated (useful for updates in tensorboard)
do_gradient_clipping = False
clipping_value = 50

##### Testing #####
do_connected_component_analysis = False
do_filter_small_components = False
min_number_of_voxels = 15
batch_size_test = 1
summaries_per_case = 10
write_probabilities = False

##### Data #####
num_channels = 3
num_slices = 1
num_classes_seg = 2  #the number of classes including the background
num_dimensions = 3
#has to be smaller than the target size
train_dim = 128
train_input_shape = [train_dim, train_dim, num_channels]
train_label_shape = [train_dim, train_dim, num_classes_seg]
test_dim = 256
test_data_shape = [test_dim, test_dim, num_channels]
test_label_shape = [test_dim, test_dim, num_classes_seg]

if num_dimensions == 3:
    num_slices_train = 4 # should be divisible by two
    train_input_shape = [num_slices_train, train_dim, train_dim, num_channels]
    train_label_shape = [num_slices_train, train_dim, train_dim, num_classes_seg]

dtype = tf.float32 #the datatype to use inside of tensorflow
dtype_np = np.float32 # the datatype used in numpy, should be the same as in tf
data_train_split = 0.75
number_of_vald = 2

##### Loader #####
vald_reader_instances = 1
file_name_capacity = 140
file_name_capacity_valid = file_name_capacity // 10
batch_capacity_valid = batch_capacity_train // 2
normalizing_method = NORMALIZING.PERCENT5

# Sample Mining
patch_shift_factor = 3  # 3*std is 99th percentile
in_between_slice_factor = 2
min_n_samples = 10
random_sampling_mode = SAMPLINGMODES.CONSTRAINED_LABEL
percent_of_object_samples = 50 #how many samples should contain the objects (in percent of samples_per_volume)
samples_per_volume = 80
samples_per_slice_object = 2
samples_per_slice_lesion = 4
samples_per_slice_bkg = 1
samples_per_slice_uni = 1
do_flip_coronal = False
do_flip_sagittal = False
do_variate_intensities = False
intensity_variation_interval = 0 #0.01
do_deform = False
deform_sigma = 10  # standard deviation of the normal distribution
points = 3  # size of the grid (3x3 grid)
add_noise = True
noise_typ = NOISETYP.GAUSSIAN
standard_deviation = 0.025
mean_poisson = 30 # relative to full scale

# Resampling
do_resampling = False
adapt_resolution = True
if adapt_resolution:
    target_spacing = [1, 1, 3]
    target_size = [256, 256] # these can be different from the network dimensions, because patches are used
target_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)  # make sure all images are oriented equally
target_type_image = sitk.sitkFloat32
target_type_label = sitk.sitkUInt8
data_background_value = 0 #data outside the image is set to this value
label_background_value = 0 #labels to this
max_rotation = 0.07  #the maximum amount of rotation that is allowed rotation will be between -pi*max_rotation and pi*max_rotation

# Weighted CE
basis_factor = 5
tissue_factor = 5
contour_factor = 2
max_weight = 1.2
tissue_threshold = -0.9

# Preprocessing
#values between outside of the quantiles norm_min_q and norm_max_q are normalized to interval [-1, 1]
#values outside this area are truncated
norm_min_q = 0.01
norm_max_q = 0.99

norm_min_v = -150
norm_max_v = 275
norm_eps = 1e-5