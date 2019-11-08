"""!
@file config.py
Sets the parameters for configuration
"""
import tensorflow as tf
import SimpleITK as sitk
import socket
from enum import Enum

##### Enums #####


class SAMPLINGMODES(Enum):
    UNIFORM = 0
    CONSTRAINED_MUSTD = 1
    CONSTRAINED_LABEL = 2


class NORMALIZING(Enum):
    WINDOW = 0
    MEAN_STD = 1
    PERCENT5 = 2


class ORGANS(Enum):
    LIVER = 'liver'
    TUMORS = 'tumors'
    LIVERANDTUMORS = 'liver+tumors'

sampe_file_name_prefix = 'ct-volume-'
label_file_name_prefix = 'vessel-segmentation-'

##### Mode #####
VERBOSE = True

if socket.gethostname() == 'ckm4cad':
    ONSERVER = True
    op_parallelism_threads = 6
    batch_size_train = 16
    training_epochs = 20
    batch_capacity_train = 4000
    train_reader_instances = 2

else:
    ONSERVER = False
    op_parallelism_threads = 3
    batch_size_train = 4
    training_epochs = 1
    batch_capacity_train = 400
    train_reader_instances = 1

summary_steps_per_epoch = 5

##### Testing #####
do_connected_component_analysis = True
batch_size_test = 1
summaries_per_case = 10

##### Data #####
organ = ORGANS.LIVER
num_channels = 3
num_slices = 1
num_classes_seg = 3
num_files = -1
train_dim = 256
train_input_shape = [train_dim, train_dim, num_channels]
train_label_shape = [train_dim, train_dim, num_classes_seg]
test_dim = 512
test_data_shape = [test_dim, test_dim, num_channels]
test_label_shape = [test_dim, test_dim, num_classes_seg]

dtype = tf.float32
data_train_split = 0.75
number_of_vald = 2

##### Loader #####
vald_reader_instances = 1
file_name_capacity = 140
file_name_capacity_valid = file_name_capacity // 10
batch_capacity_valid = batch_capacity_train // 2
normalizing_method = NORMALIZING.WINDOW

# Sample Mining
patch_shift_factor = 3  # 3*std is 99th percentile
in_between_slice_factor = 2
min_n_samples = 10
random_sampling_mode = SAMPLINGMODES.CONSTRAINED_LABEL
percent_of_object_samples = 50  # %
samples_per_volume = 40
samples_per_slice_object = 2
samples_per_slice_lesion = 4
samples_per_slice_bkg = 1
samples_per_slice_uni = 1
do_flip_coronal = False
do_flip_sagittal = False
do_variate_intensities = True
intensity_variation_interval = 0.01

# Resampling
adapt_resolution = True
if adapt_resolution:
    target_spacing = [0.75, 0.75, 1.5]
    target_size = [512, 512]
target_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)  # make sure all images are oriented equally
target_type_image = sitk.sitkFloat32
target_type_label = sitk.sitkUInt8
data_background_value = -1000
label_background_value = 0
max_rotation = 0.07

# Tversky
tversky_alpha = 0.3
tversky_beta = 1 - tversky_alpha

# Weighted CE
basis_factor = 5
tissue_factor = 5
contour_factor = 2
max_weight = 1.2
tissue_threshold = -0.9

# Preprocessing
norm_min_v = -150
norm_max_v = 350
norm_eps = 1e-5

##### Network #####
sparse_cardinality = 2