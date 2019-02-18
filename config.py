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
    CONSTRAINED_LABEL = 1

##### Mode #####
VERBOSE = True

if socket.gethostname() == 'ckm4cad':
    ONSERVER = True
    op_parallelism_threads = 6
    batch_size = 16
    training_epochs = 20
    batch_capacity = 4000
    train_reader_instances = 2

else:
    ONSERVER = False
    op_parallelism_threads = 3
    batch_size = 8
    training_epochs = 1
    batch_capacity = 400
    train_reader_instances = 1

write_step = 2500
summary_step = 100

##### Testing #####
do_connected_component_analysis = True
test_size = 1
summaries_per_case = 10

##### Data #####
data_train_split = 0.75
number_of_vald = 2
dtype = tf.float32
num_channels = 3
num_classes_seg = 2
train_dim = 256
train_input_shape = [train_dim, train_dim, num_channels]
train_label_shape = [train_dim, train_dim, num_classes_seg]
test_dim = 512
test_data_shape = [test_dim, test_dim, num_channels]
test_label_shape_seg = [test_dim, test_dim, num_classes_seg]

##### Loader #####
vald_reader_instances = 1
file_name_capacity = 140
batch_capacity_valid = batch_capacity//2

# Sample Mining
percent_of_object_samples = 50  # %
samples_per_slice_liver = 2
samples_per_slice_lesion = 4
samples_per_slice_bkg = 1
samples_per_slice_uni = 1
patch_shift_factor = 3  # 3*std is 99th percentile
in_between_slice_factor = 2
slice_shift = ((num_channels - 1) // 2) * in_between_slice_factor
min_n_samples = 10
random_sampling_mode = SAMPLINGMODES.CONSTRAINED_MUSTD

# Resampling
adapt_resolution = True
if adapt_resolution:
    target_spacing = [0.75, 0.75, 1.25]
    target_size = [512, 512]
target_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)  # make sure all images are oriented equally
target_type_image = sitk.sitkFloat32
target_type_label = sitk.sitkUInt32
data_background_value = -1000
label_background_value = 0
max_rotation = 0.07

# Preprocessing
norm_min_v = -200
norm_max_v = 250
norm_eps = 1e-5

# Tversky
tversky_alpha = 0.3
tversky_beta = 1 - tversky_alpha

# Weighted CE
basis_factor = 5
tissue_factor = 5
contour_factor = 2
max_weight = 1.2
tissue_threshold = -0.9