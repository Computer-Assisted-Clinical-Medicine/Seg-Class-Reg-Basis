# %% [markdown]
"""
# Profile Seg_data_loader
## Imports and Definitions

This file is to test the timing of the different data loaders. The functions are
also profiled.
"""

# pylint: disable=pointless-string-statement, protected-access

import cProfile
import os
import pstats
import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from IPython.display import display

from . import config as cfg
from .test_seg_data_loader import (
    get_loader,
    load_dataset,
    set_parameters_according_to_dimension,
    set_seeds,
)

# suppress tensorflow output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # pylint: disable=unused-import, wrong-import-order, wrong-import-position

SHOW_PLOTS = False


def time_functions(dimension, name, timing_result):
    """
    Time the individual functions in the data loaders
    """

    test_dir = Path("test_data")

    set_parameters_according_to_dimension(dimension, 2, test_dir)

    cfg.sample_target_spacing = (0.5, 0.5, None)

    set_seeds()

    # get names from csv
    _, files_list_b, file_dict = load_dataset(test_dir)

    # generate loader
    data_loader = get_loader(name, file_dict, 0.4)

    # time the individual functions
    load_time = []
    sample_time = []
    augmentation_images_time = []
    augmentation_numpy_time = []
    convert_images_time = []

    for file_id in files_list_b:
        start_time = time.perf_counter()
        data, lbl = data_loader._load_file(file_id)
        load = time.perf_counter()
        load_time.append(load - start_time)

        _, _ = data_loader._get_samples_from_volume(data, lbl)
        get_samples = time.perf_counter()
        sample_time.append(get_samples - load)

        # time augmentation
        if hasattr(data_loader, "_augment_numpy"):
            # augment whole images
            _, _ = data_loader._augment_images(data, lbl)

            augment_1 = time.perf_counter()
            augmentation_images_time.append(augment_1 - get_samples)
            # convert samples to numpy arrays
            data = sitk.GetArrayFromImage(data)
            lbl = sitk.GetArrayFromImage(lbl)
            converted = time.perf_counter()
            convert_images_time.append(converted - augment_1)
            # augment the numpy arrays
            _, _ = data_loader._augment_numpy(data, lbl)
            augment_2 = time.perf_counter()
            augmentation_numpy_time.append(augment_2 - converted)

    print(
        f"\tExecution time for {name} {dimension}D: load: {np.mean(load_time):.2f}s"
        + f", sample (incl Augm): {np.mean(sample_time):.2f}s"
    )
    timing_dict = {
        "load file": np.mean(load_time),
        "get sample": np.mean(sample_time),
    }
    if len(augmentation_images_time) > 0:
        print(
            f"\tAugm im: {np.mean(augmentation_images_time):.2f}s, Augm. np: {np.mean(augmentation_numpy_time):.2f}s"
            + f", Conv im: {np.mean(convert_images_time):.2f}"
        )
        timing_dict = {
            "augment img.": np.mean(augmentation_images_time),
            "augment np": np.mean(augmentation_numpy_time),
            "conv. img.": np.mean(convert_images_time),
            **timing_dict,
        }

    timing_result[f"{name}-{dimension}D"] = timing_dict

    return timing_result


def profile_functions(dimension, name):
    """
    Generate a profile for the individual functions
    """

    test_dir = Path("test_data")

    set_parameters_according_to_dimension(dimension, 2, test_dir)

    profile_dir = test_dir / "profiles"
    if not profile_dir.exists():
        profile_dir.mkdir()
    profile_file = profile_dir / f"{name}-{dimension}D.prof"

    set_seeds()

    # get names from csv
    _, files_list_b, file_dict = load_dataset(test_dir)

    # generate loader
    data_loader = get_loader(name, file_dict, 0.4)

    def load_all_files():
        for file_id in files_list_b:
            data, lbl = data_loader._load_file(file_id)
            _, _ = data_loader._get_samples_from_volume(data, lbl)

    # profile the function
    profiler = cProfile.Profile()
    profiler.enable()
    load_all_files()
    profiler.disable()
    # dump stats file
    profiler.dump_stats(profile_file)
    profiler_stats = pstats.Stats(profiler)
    profiler_stats.sort_stats(pstats.SortKey.CUMULATIVE)
    profiler_stats.print_stats(15)

    return profile_file


def time_wrapper(dimension, name, timing_result):
    """
    Wrapper used to time the different loaders
    """
    n_epochs = 1

    test_dir = Path("test_data")

    set_parameters_according_to_dimension(dimension, 2, test_dir)

    set_seeds()

    # get names from csv
    file_list, _, file_dict = load_dataset(test_dir)

    # generate loader
    data_loader = get_loader(name, file_dict, 0.4)

    data_loader.get_filenames(str(file_list[0]))

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
            batch_size=cfg.batch_size_train,
            read_threads=cfg.vald_reader_instances,
        )
    else:
        dataset = data_loader(
            file_list,
            batch_size=cfg.batch_size_train,
            read_threads=cfg.vald_reader_instances,
        )

    counter = 0
    load_time = []

    start_time = time.perf_counter()

    for sample in dataset:
        if counter == 0:
            setup_time = time.perf_counter() - start_time
        else:
            load_time.append(time.perf_counter() - start_time)

        if SHOW_PLOTS:
            if name == "train":
                # convert to numpy
                x_t, y_t = sample[0].numpy(), sample[1].numpy()
                plot(dimension, samples_lbl=x_t, labels_lbl=y_t)

        # print(counter)
        counter += 1

        # get time to exclude the checks
        start_time = time.perf_counter()

    assert counter != 0

    if name != "test":
        assert counter == cfg.samples_per_volume * cfg.num_files // cfg.batch_size_train

    if len(load_time) == 0:
        load_time = [setup_time]

    print(
        f"\tExecution time for one step: {np.mean(load_time):.2f}s ({np.sum(load_time):.2f}s total)"
    )
    print(f"\tSetup time: {setup_time:.2f}s")
    # add to dict
    time_name = f"{name}-{dimension}D"
    timing_result[time_name]["step"] = np.mean(load_time)
    timing_result[time_name]["setup"] = setup_time
    timing_result[time_name]["total"] = np.sum(load_time)

    return timing_result


def plot(dimension, samples_lbl, labels_lbl, samples_bkr=None, labels_bkr=None):
    """
    Plot a histogram of the foreground samples and labels
    """
    plt.hist(samples_lbl.reshape(-1))
    plt.title("Histogram of foreground samples")
    plt.show()
    plt.close()

    plt.hist(labels_lbl.reshape(-1))
    plt.title("Histogram of foreground labels")
    plt.show()
    plt.close()

    nsamples = 5
    ncols = samples_lbl.shape[-1] + 1
    nrows = nsamples
    indices = np.sort(np.random.choice(np.arange(samples_lbl.shape[0]), nsamples))
    _, axes = plt.subplots(nrows, ncols, figsize=(11, 9))

    for ax_r, sample_r, label in zip(axes, samples_lbl[indices], labels_lbl[indices]):
        index = np.random.choice(np.arange(sample_r.shape[0]))
        for ax, sample in zip(ax_r[:-1], np.moveaxis(sample_r, -1, 0)):
            if dimension == 3:
                ax.imshow(sample[index])
            else:
                ax.imshow(sample)
        if dimension == 3:
            ax_r[-1].imshow(label[index, ..., :-1], vmin=0, vmax=1)
        else:
            ax_r[-1].imshow(label[..., :-1], vmin=0, vmax=1)

    plt.tight_layout()
    plt.show()
    plt.close()


# %% [markdown]
"""
## Evaluate the timing
"""

timing: Dict[str, float] = {}

dimensions = [2, 3]
names = ["train"]

# call functions and time them
for dim in dimensions:
    for NAME in names:
        print(f"{NAME} {dim}D:")
        time_functions(dim, NAME, timing)
        time_wrapper(dim, NAME, timing)

timing_pd = pd.DataFrame(timing).T
# set index
timing_pd.set_index(
    pd.MultiIndex.from_tuples(tuple(timing_pd.index.str.split("-"))), inplace=True
)
display(timing_pd.round(3))

# %% [markdown]
"""
## Analyze the profiles
"""

profile_files = {}

NAME = "train"

# call functions and time them
for dim in dimensions:
    print(f"{NAME} {dim}D:")

    # profile the individual functions
    t_name = f"{NAME}-{dim}D"
    profile_files[t_name] = profile_functions(dim, NAME)

print("For a graphical profile call:")
for NAME, path in profile_files.items():
    print(NAME)
    print(f"\tsnakeviz {path}")
