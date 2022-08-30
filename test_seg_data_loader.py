"""
Test the different data loaders using different settings
"""

from pathlib import Path

import GPUtil
import numpy as np
import pytest
import SimpleITK as sitk
import tensorflow as tf

from . import create_test_files
from . import config as cfg
from .segloader import SegLoader
from .segapplyloader import ApplyLoader

# pylint: disable=protected-access,duplicate-code


def set_seeds():
    tf.keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)


def get_loader(name, file_dict, percent_obj=0.33):
    """
    Get the data loader using the specified module, normalization method and
    object percentage.
    """
    # generate loader
    if name == "train":
        data_loader = SegLoader(
            name="training_loader", frac_obj=percent_obj, file_dict=file_dict
        )
    elif name == "vald":
        data_loader = SegLoader(
            mode=SegLoader.MODES.VALIDATE,
            frac_obj=percent_obj,
            name="validation_loader",
            file_dict=file_dict,
        )
    elif name == "test":
        data_loader = ApplyLoader(
            mode=SegLoader.MODES.APPLY,
            name="test_loader",
            file_dict=file_dict,
        )
    return data_loader


def load_dataset(test_dir, n_classification=0, n_regression=0, load_labels=True):
    """
    Load the dataset from the test directory and convert the file lists to the
    right types. Some classification and regression labels can also be added,
    the classification labels will be a random one-hot encoded vector between 2
    and 20 entries and the regression values will be random between 0 and 1
    """
    # add data path
    file_list = create_test_files.create_test_files(test_dir)

    if n_classification > 0:
        class_shapes = [np.random.randint(2, 20) for _ in range(n_classification)]

    file_dict = {}
    for f in file_list:
        f_path = Path(f)
        name = f_path.name
        image_name = f_path.parent / (cfg.sample_file_name_prefix + name + cfg.file_suffix)
        labels_name = f_path.parent / (cfg.label_file_name_prefix + name + cfg.file_suffix)
        file_dict[name] = {"image": image_name.relative_to(test_dir)}
        if load_labels:
            file_dict[name]["labels"] = labels_name.relative_to(test_dir)
        if n_classification > 0:
            file_dict[name]["classification"] = [
                np.eye(c)[np.random.randint(c)] for c in class_shapes
            ]
        if n_regression > 0:
            file_dict[name]["regression"] = [
                np.random.uniform() for _ in range(n_regression)
            ]

    # only use names for the file list
    file_list = list(file_dict.keys())

    id_tensor = tf.squeeze(tf.convert_to_tensor(file_list, dtype=tf.string))
    # Create dataset from list of file names
    file_list_ds = tf.data.Dataset.from_tensor_slices(id_tensor)
    # convert it back (for the right types)
    files_list_b = list(file_list_ds.as_numpy_iterator())

    cfg.num_files = len(file_list)

    return file_list, files_list_b, file_dict


def set_parameters_according_to_dimension(
    dimension, num_channels, preprocessed_dir, a_name="UNet"
):
    """This function will set up the shapes in the cfg module so that they
    will run on the current GPU.
    """

    cfg.number_of_vald = 2

    cfg.num_channels = num_channels
    cfg.train_dim = 128  # the resolution in plane
    cfg.num_slices_train = 32  # the resolution in z-direction

    # determine batch size
    cfg.batch_size_train = estimate_batch_size(dimension, a_name)
    cfg.batch_size_valid = cfg.batch_size_train

    # set shape according to the dimension
    if dimension == 2:
        # set shape
        cfg.train_input_shape = [cfg.train_dim, cfg.train_dim, cfg.num_channels]
        cfg.train_label_shape = [cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]

        # set sample numbers
        # there are 10-30 layers per image containing foreground data. Half the
        # samples are taken from the foreground, so take about 64 samples
        # to cover all the foreground pixels at least once on average, but
        cfg.samples_per_volume = 64
        cfg.batch_capacity_train = (
            4 * cfg.samples_per_volume
        )  # chosen as multiple of samples per volume

    elif dimension == 3:
        # set shape
        # if batch size too small, decrease z-extent
        if cfg.batch_size_train < 4:
            cfg.num_slices_train = cfg.num_slices_train // 2
            cfg.batch_size_train = cfg.batch_size_train * 2
            # if still to small, decrease patch extent in plane
            if cfg.batch_size_train < 4:
                cfg.train_dim = cfg.train_dim // 2
                cfg.batch_size_train = cfg.batch_size_train * 2
        cfg.train_input_shape = [
            cfg.num_slices_train,
            cfg.train_dim,
            cfg.train_dim,
            cfg.num_channels,
        ]
        cfg.train_label_shape = [
            cfg.num_slices_train,
            cfg.train_dim,
            cfg.train_dim,
            cfg.num_classes_seg,
        ]

        # set sample numbers
        # most patches should cover the whole tumor, so a lower sample number
        # can be used
        cfg.samples_per_volume = 8
        cfg.batch_capacity_train = (
            4 * cfg.samples_per_volume
        )  # chosen as multiple of samples per volume

    # set the valid batch size
    cfg.batch_size_valid = cfg.batch_size_train
    # see if the batch size is bigger than the validation set
    if cfg.samples_per_volume * cfg.number_of_vald <= cfg.batch_size_valid:
        cfg.batch_size_valid = cfg.samples_per_volume * cfg.number_of_vald

    # set config
    if not preprocessed_dir.exists():
        print(f"{preprocessed_dir} not found.")
    cfg.data_base_dir = str(preprocessed_dir)


def estimate_batch_size(dimension, a_name):
    """The batch size estimation is basically trail and error. So far tested
    with 128x128x2 patches in 2D and 128x128x32x2 in 3D, if using different
    values, guesstimate the relation to the memory.

    Returns
    -------
    int
        The recommended batch size
    """
    # set batch size
    # determine GPU memory (in MB)
    gpu_number = int(tf.test.gpu_device_name()[-1])
    gpu_memory = int(np.round(GPUtil.getGPUs()[gpu_number].memoryTotal))

    if a_name == "UNet":
        # filters scale after the first filter, so use that for estimation
        first_f = 8
        if dimension == 2:
            # this was determined by trail and error for 128x128x2 patches
            memory_consumption_guess = 4 * first_f
        elif dimension == 3:
            # this was determined by trail and error for 128x128x32x2 patches
            memory_consumption_guess = 128 * first_f
    elif a_name == "DenseTiramisu":
        if dimension == 2:
            memory_consumption_guess = 1024
        if dimension == 3:
            memory_consumption_guess = 4096
    elif a_name == "DeepLabv3plus":
        if dimension == 2:
            memory_consumption_guess = 1024
        if dimension == 3:
            raise NotImplementedError(f"3D not implemented for {a_name}.")
    else:
        raise NotImplementedError(f"No heuristic implemented for {a_name}.")

    # return estimated recommended batch number
    return np.round(gpu_memory // memory_consumption_guess)


dimensions = [2, 3]
names = [
    "train",
    # "vald"
]
frac_objects = [0, 0.3, 1]
n_class = [0, 1, 5, 10]
n_reg = [0, 1, 5, 10]
load_labels_var = [True, False]


@pytest.mark.parametrize("dimension", dimensions)
@pytest.mark.parametrize("name", names)
@pytest.mark.parametrize("frac_obj", frac_objects)
@pytest.mark.parametrize("n_classification", n_class)
@pytest.mark.parametrize("n_regression", n_reg)
@pytest.mark.parametrize("load_labels", load_labels_var)
def test_functions(dimension, name, frac_obj, n_classification, n_regression, load_labels):
    """Test the individual functions contained in the wrapper.

    Parameters
    ----------
    dimension : int
        The dimension (2 or 3)
    name : str
        The name, train, test or vald
    """

    test_dir = Path("test_data")

    set_parameters_according_to_dimension(dimension, 2, test_dir)

    set_seeds()

    # get names from csv
    file_list, files_list_b, file_dict = load_dataset(
        test_dir=test_dir,
        n_classification=n_classification,
        n_regression=n_regression,
        load_labels=load_labels,
    )

    # generate loader
    data_loader = get_loader(name, file_dict, frac_obj)

    print(f"Loading Dataset {name}.")

    print("\tLoad a numpy sample")
    data_read = data_loader._read_file_and_return_numpy_samples(files_list_b[0])

    assert (
        data_read[0].shape[0] == cfg.samples_per_volume
    ), "Wrong number of samples per volume"
    assert data_read[0].shape[1:] == tuple(cfg.train_input_shape), "Wrong sample shape"

    assert (
        data_read[0].shape[0] == cfg.samples_per_volume
    ), "Wrong number of samples per volume"
    if load_labels:
        assert data_read[1].shape[1:] == tuple(cfg.train_label_shape), "Wrong label shape"

    if name == "test":
        samples = data_read[0]
    else:
        if n_classification == 0 and n_regression == 0 and load_labels:
            samples, labels = data_read
            class_reg = None
        if n_classification == 0 and n_regression == 0 and not load_labels:
            samples = data_read[0]
            class_reg = None
        elif load_labels:
            samples, labels, class_reg = data_read[0], data_read[1], data_read[2:]
        else:
            samples, class_reg = data_read[0], data_read[1:]

        print(f"\tSamples from foreground shape: {samples.shape}")
        if load_labels:
            print(f"\tLabels from foreground shape: {labels.shape}")

            assert samples.shape[:-1] == labels.shape[:-1]

            nan_slices = np.all(np.isnan(labels), axis=(1, 2, 3))
            assert not np.any(nan_slices), f"{nan_slices.sum()} label slices contain NANs"

        if class_reg is not None:
            # check the shapes of the samples
            assert len(class_reg) == n_classification + n_regression
            if n_classification > 0:
                first = file_dict[file_list[0]]["classification"]
                required_shape = [(cfg.samples_per_volume,) + f.shape for f in first]
                sample_shape = [s.shape for s in class_reg[:n_classification]]
                assert np.all([r == s for r, s in zip(required_shape, sample_shape)])
            if n_regression > 0:
                reg_shape = (cfg.samples_per_volume, 1)
                assert np.all([r.shape == reg_shape for r in class_reg[-n_regression:]])

    nan_slices = np.all(np.isnan(samples), axis=(1, 2, 3))
    assert not np.any(nan_slices), f"{nan_slices.sum()} sample slices contain NANs"

    # call the wrapper function
    data_loader._read_wrapper(
        id_data_set=tf.squeeze(tf.convert_to_tensor(file_list[0], dtype=tf.string))
    )


@pytest.mark.parametrize("dimension", dimensions)
@pytest.mark.parametrize("name", names)
@pytest.mark.parametrize("frac_obj", frac_objects)
@pytest.mark.parametrize("n_classification", n_class)
@pytest.mark.parametrize("n_regression", n_reg)
def test_wrapper(dimension, name, frac_obj, n_classification, n_regression):
    """Test the complete wrapper and check shapes

    Parameters
    ----------
    dimension : int
        The dimension (2 or 3)
    name : str
        The name, train, test or vald

    Raises
    ------
    Exception
        Error as detected
    """
    n_epochs = 1

    test_dir = Path("test_data")

    set_parameters_according_to_dimension(dimension, 2, test_dir)

    set_seeds()

    # get names from csv
    file_list, _, file_dict = load_dataset(
        test_dir=test_dir,
        n_classification=n_classification,
        n_regression=n_regression,
    )

    # generate loader
    data_loader = get_loader(name, file_dict, frac_obj)

    data_file, _ = data_loader.get_filenames(str(file_list[0]))
    first_image = sitk.GetArrayFromImage(sitk.ReadImage(data_file))

    print(f"Loading Dataset {name}.")

    # call the loader
    if name == "train":
        dataset = data_loader(
            file_list,
            batch_size=cfg.batch_size_train,
            n_epochs=n_epochs,
            read_threads=cfg.train_reader_instances,
        )
    elif name == "test":
        dataset = data_loader(
            file_list[0],  # only pass one file to the test loader
        )
    else:
        dataset = data_loader(
            file_list,
            batch_size=cfg.batch_size_train,
            read_threads=cfg.vald_reader_instances,
            n_epochs=n_epochs,
        )

    print("\tLoad samples using the data loader")

    # count iterations
    counter = 0
    # save fraction of slices with samples
    n_objects = []
    n_background = []

    for sample in dataset:
        # test set only contains samples, not labels
        if name == "test":
            x_t = sample
        else:
            if n_classification == 0 and n_regression == 0:
                x_t, y_t = sample
                cr_t = None
            else:
                x_t, y_t, cr_t = sample[0], sample[1][0], sample[1][1:]

        # check shape
        if name != "test":
            assert cfg.batch_size_train == x_t.shape[0]
            assert cfg.num_channels == x_t.shape[-1]
            assert cfg.batch_size_train == y_t.shape[0]
            if cr_t is not None:
                assert len(cr_t) == n_classification + n_regression
                if n_classification > 0:
                    first = file_dict[file_list[0]]["classification"]
                    required_shape = [(cfg.batch_size_train,) + f.shape for f in first]
                    sample_shape = [s.shape for s in cr_t[:n_classification]]
                    assert np.all([r == s for r, s in zip(required_shape, sample_shape)])
                if n_regression > 0:
                    reg_shape = (cfg.batch_size_train, 1)
                    assert np.all([r.shape == reg_shape for r in cr_t[-n_regression:]])
        else:
            if dimension == 2:
                assert first_image.shape[0] == x_t.numpy().shape[0]

        # look for nans
        nan_slices = np.all(np.isnan(x_t.numpy()), axis=(1, 2, 3))
        if np.any(nan_slices):
            print(f"{nan_slices.sum()} sample slices only contain NANs")

        if name != "test":
            nan_slices = np.all(np.isnan(y_t.numpy()), axis=(1, 2, 3))
            assert not np.any(
                nan_slices
            ), f"{nan_slices.sum()} label slices only contain NANs"

            # check that the labels are always one
            assert np.all(np.sum(y_t.numpy(), axis=-1) == 1)

            nan_frac = np.mean(np.isnan(x_t.numpy()))
            assert (
                nan_frac < 0.01
            ), f"More than 2% nans in the image ({int(nan_frac*100)}%)."

            # check for labels in the slices
            if dimension == 3:
                n_bkr_per_sample = np.sum(y_t.numpy()[..., 0], axis=(1, 2, 3)).astype(int)
                n_object_per_sample = np.sum(y_t.numpy()[..., 1], axis=(1, 2, 3)).astype(
                    int
                )
            else:
                n_bkr_per_sample = np.sum(y_t.numpy()[..., 0], axis=(1, 2)).astype(int)
                n_object_per_sample = np.sum(y_t.numpy()[..., 1], axis=(1, 2)).astype(int)
            if np.all(n_object_per_sample == 0):
                raise Exception(
                    "All labels are zero, no objects were found, either the labels are incorrect "
                    + "or there was a problem processing the image."
                )

            n_objects.append(n_object_per_sample)
            n_background.append(n_bkr_per_sample)

        # print(counter)
        counter += 1

    # there should be at least one iteration
    assert counter != 0

    # test that the number of samples per epoch is correct
    if name != "test":
        assert counter == cfg.samples_per_volume * cfg.num_files // cfg.batch_size_train

        # check the fraction of objects per sample
        n_objects = np.array(n_objects)
        n_background = np.array(n_background)

        # get the fraction of samples containing a label
        assert np.mean(n_objects.reshape(-1) > 0) >= frac_obj


def test_apply_loader():
    """
    Test the apply loader by loading a few images and verify the padding
    """

    test_dir = Path("test_data")

    set_parameters_according_to_dimension(3, 2, test_dir)

    # get names from csv
    file_list, _, file_dict = load_dataset(test_dir)
    filename = file_list[0]

    loader = get_loader("test", file_dict)

    image_data = loader(filename)

    # make sure the processed image and the image without padding are the same (except the batch dimension)
    processed_image = sitk.GetArrayFromImage(loader.get_processed_image(filename))
    padding_removed = loader.remove_padding(image_data)
    assert np.allclose(
        processed_image, padding_removed[0]
    ), "image with padding removed is not the same as original"

    # test the stitching
    window_shape = np.array(cfg.train_input_shape[:3]) + 4
    # turn an image into windows and then stitch it back together
    image_data_windowed = loader.get_windowed_test_sample(image_data, window_shape)
    image_data_stitched = loader.stitch_patches(image_data_windowed)
    image_data_stitched_no_padding = loader.remove_padding(image_data_stitched)
    assert np.allclose(padding_removed, image_data_stitched_no_padding)


if __name__ == "__main__":
    # run functions for better debugging
    for dim in dimensions:
        for mod_name in names:
            for n_c in n_class:
                for n_r in n_reg:
                    for ll in load_labels_var:
                        if ll:
                            FR_OBJ = 0.4
                        else:
                            FR_OBJ = 0
                        test_functions(
                            dim,
                            mod_name,
                            frac_obj=FR_OBJ,
                            n_classification=n_c,
                            n_regression=n_r,
                            load_labels=ll,
                        )
                        test_wrapper(
                            dim,
                            mod_name,
                            frac_obj=FR_OBJ,
                            n_classification=n_c,
                            n_regression=n_r,
                        )
