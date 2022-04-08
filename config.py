"""!
@file config.py
Sets the parameters for configuration
"""
import socket
from typing import Any, Optional

import numpy as np
import tensorflow as tf

# pylint: disable=invalid-name

##### Names and Paths #####

# these files are used to store the different sets
train_csv = "train.csv"
fine_csv = "fine.csv"
vald_csv = "vald.csv"
test_csv = "test.csv"

# prefixes are used for file names
sample_file_name_prefix = "sample-"
label_file_name_prefix = "label-"
# the suffix determines the format
file_suffix = ".nii.gz"
# preprocessed_dir
data_base_dir: Any = None


##### Shapes and Capacities #####
if socket.gethostname() == "ckm4cad":
    ONSERVER = True
    batch_size_train = 16
    batch_capacity_train = 4000
    train_reader_instances = 2

else:
    ONSERVER = False
    batch_size_train = 4
    batch_capacity_train = 400
    train_reader_instances = 1

batch_size_valid = batch_size_train
vald_reader_instances = 1


##### Data #####
num_channels = 3
num_slices = 1
num_classes_seg = 2  # the number of classes including the background
num_dimensions = 3
# has to be smaller than the target size
train_dim = 128

if num_dimensions == 2:
    train_input_shape = [train_dim, train_dim, num_channels]
    train_label_shape = [train_dim, train_dim, num_classes_seg]
elif num_dimensions == 3:
    num_slices_train = 16  # should be divisible by 16 for UNet
    train_input_shape = [num_slices_train, train_dim, train_dim, num_channels]
    train_label_shape = [num_slices_train, train_dim, train_dim, num_classes_seg]

dtype = tf.float32  # the datatype to use inside of tensorflow
dtype_np = np.float32  # the datatype used in numpy, should be the same as in tf
data_train_split = 0.75
number_of_vald = 4

###### Sample Mining #####
percent_of_object_samples = (
    0.5  # how many samples should contain the objects (in percent of samples_per_volume)
)
samples_per_volume = 80  # the number of sample per image
background_label_percentage = (
    0.15  # the maximum fraction of labelled voxels allowed in a background patch
)
# This can be used to avoid oversampling large tumors.

add_noise = False
noise_typ = None  # TODO: add rician noise
standard_deviation = 0.025
mean_poisson = 30  # relative to full scale

max_rotation = 0.0  # the maximum amount of rotation that is allowed (between 0 and 1)
# resolution is augmented by a factor between min_resolution_augment and max_resolution_augment
# the values can be scalars or lists, if a list is used, then all axes are scaled individually
min_resolution_augment = 1
max_resolution_augment = 1
sample_target_spacing = [1, 1, 3]

##### Other variables #####
num_files: Optional[int] = None  # TODO: remove from config
