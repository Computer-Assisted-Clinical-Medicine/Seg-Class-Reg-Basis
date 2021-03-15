"""!
@file config.py
Sets the parameters for configuration
"""
import socket

import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from .segbasisloader import NOISETYP, NORMALIZING


##### Names and Paths #####

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
# preprocessed_dir
preprocessed_dir = 'preprocessed'


##### Shapes and Capacities #####
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

batch_size_valid = batch_size_train
vald_reader_instances = 1
file_name_capacity = 140


##### Data #####
num_channels = 3
num_slices = 1
num_classes_seg = 2  #the number of classes including the background
num_dimensions = 3
#has to be smaller than the target size
train_dim = 128

if num_dimensions == 2:
    train_input_shape = [train_dim, train_dim, num_channels]
    train_label_shape = [train_dim, train_dim, num_classes_seg]
elif num_dimensions == 3:
    num_slices_train = 16 # should be divisible by 16
    train_input_shape = [num_slices_train, train_dim, train_dim, num_channels]
    train_label_shape = [num_slices_train, train_dim, train_dim, num_classes_seg]

dtype = tf.float32 #the datatype to use inside of tensorflow
dtype_np = np.float32 # the datatype used in numpy, should be the same as in tf
data_train_split = 0.75
number_of_vald = 4


##### Preprocessing #####
normalizing_method = NORMALIZING.PERCENT5
#values between outside of the quantiles norm_min_q and norm_max_q are normalized to interval [-1, 1]
norm_min_q = 0.01
norm_max_q = 0.99
#values between outside of norm_min_v and norm_max_v are normalized to interval [-1, 1]
norm_min_v = -150
norm_max_v = 275

do_resampling = True
if do_resampling:
    target_spacing = [1, 1, 3]
    target_size = [256, 256] # TODO: remove
target_type_image = sitk.sitkFloat32
target_type_label = sitk.sitkUInt8
data_background_value = 0 #data outside the image is set to this value
label_background_value = 0 #labels to this

###### Sample Mining #####
percent_of_object_samples = 50 #how many samples should contain the objects (in percent of samples_per_volume)
samples_per_volume = 80

add_noise = True
noise_typ = NOISETYP.GAUSSIAN
standard_deviation = 0.025
mean_poisson = 30 # relative to full scale

max_rotation = 0.07  # the maximum amount of rotation that is allowed rotation will be between -pi*max_rotation and pi*max_rotation
# resolution is augmented by a factor between min_resolution_augment and max_resolution_augment
# the values can be scalars or lists, if a list is used, then all axes are scaled individually 
min_resolution_augment = 0.98
max_resolution_augment = 1.02

do_deform = False # TODO: implement with random spline field
deform_sigma = 10  # standard deviation of the normal distribution
points = 3  # size of the grid (3x3 grid)


##### Testing #####
write_probabilities = False #TODO: implement


##### Loss Setting #####

# Weighted CE
basis_factor = 5
tissue_factor = 5
contour_factor = 2
max_weight = 1.2
tissue_threshold = -0.9