"""!
@file config.py
Sets the parameters for configuration
"""
import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import socket
from enum import Enum

##### Enums #####


class SAMPLINGMODES(Enum):
    UNIFORM = 0
    CONSTRAINED_MUSTD = 1
    CONSTRAINED_LABEL = 2

##### Mode #####
VERBOSE = True

if socket.gethostname() == 'ckm4cad':
    ONSERVER = True
    batch_size = 8
    training_epochs = 20
    batch_capacity = 4000
    train_reader_instances = 2

else:
    ONSERVER = False
    batch_size = 8
    training_epochs = 200
    batch_capacity = 2000
    train_reader_instances = 2

write_step = 2500
summary_step = 50

##### Testing #####
do_connected_component_analysis = False
test_size = 1
summaries_per_case = 10

##### Data #####
num_channels = 2
num_slices = 1
num_classes_seg = 2
train_dim = 192
train_input_shape = [train_dim, train_dim, num_channels]
train_label_shape = [train_dim, train_dim, num_classes_seg]
test_dim = 240
test_data_shape = [test_dim, test_dim, num_channels]
test_label_shape_seg = [test_dim, test_dim, num_classes_seg]
dtype = tf.float32
data_train_split = 0.75
number_of_vald = 2

##### Loader #####
vald_reader_instances = 1
file_name_capacity = 11
batch_capacity_valid = batch_capacity//4
do_flip_coronal = True
do_flip_sagittal = True
use_smooth_labels = True

# Sample Mining
patch_shift_factor = 1  # 3*std is 99th percentile
in_between_slice_factor = 2
slice_shift = ((num_channels - 1) // 2) * in_between_slice_factor
min_n_samples = 10
random_sampling_mode = SAMPLINGMODES.CONSTRAINED_MUSTD
percent_of_object_samples = 50  # %
samples_per_slice_lesion = 3
samples_per_slice_bkg = 1
samples_per_slice_uni = 1

# Resampling
adapt_resolution = False
if adapt_resolution:
    target_spacing = [0.75, 0.75, 1.25]
    target_size = [512, 512]
target_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)  # make sure all images are oriented equally
target_type_image = sitk.sitkFloat32
target_type_label = sitk.sitkFloat32
data_background_value = 0
label_background_value = 0
max_rotation = 0.5 #*pi equals 90 degrees

# Tversky
tversky_alpha = 0.2
tversky_beta = 1 - tversky_alpha


# Preprocessing
norm_min_v_t1 = 500
norm_max_v_t1 = 3000
norm_min_v_t2 = 20
norm_max_v_t2 = 200
norm_min_v = norm_min_v_t2
norm_eps = 1e-5

##### Network #####
sparse_cardinality = 4
