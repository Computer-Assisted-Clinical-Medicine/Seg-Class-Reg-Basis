"""
Create test files with blobs to classify
"""
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from SegClassRegBasis import config as cfg


def create_test_files(test_path=Path("test_data"), n_files=5) -> list:
    """Create test file consisting of spheres for testing

    Parameters
    ----------
    test_path : Path, optional
        The path where the files will be written, by default Path('test_data')
    n_files : int, optional
        The number of files to create, by default 5

    Returns
    -------
    list
        List of file path in the format required by the framework
    """
    spacing = np.array([4, 1, 1, 1])

    if not test_path.exists():
        test_path.mkdir()

    test_path = test_path / f"{cfg.num_channels}_channels"
    if not test_path.exists():
        test_path.mkdir()

    # write random data to file
    training_files = []
    for i in range(n_files):
        patient_number = f"test{i}"

        label_file = (
            test_path / f"{cfg.label_file_name_prefix}{patient_number}{cfg.file_suffix}"
        )
        image_file = (
            test_path / f"{cfg.sample_file_name_prefix}{patient_number}{cfg.file_suffix}"
        )

        training_files.append(str(test_path / f"{patient_number}"))

        if label_file.exists() and image_file.exists():
            continue

        # take a random number of slices
        shape = (np.random.randint(low=20, high=60), 256, 256, cfg.num_channels)

        pos = (np.indices(shape[:3]).T * spacing[:3]).T
        center = np.array(shape[:3]) * spacing[:3] * (0.5 + (np.random.rand(3) - 0.5) * 0.4)
        # get distance to center (ignoring the channel)
        dist_to_center = np.sqrt(np.sum(np.square(center - pos[:3].T), axis=-1)).T
        dist = 1 - dist_to_center / dist_to_center.max()
        # make the circle a random size
        radius = np.random.rand() * 0.2 + 0.05
        labels = dist > 1 - radius
        assert np.sum(labels) > 0

        # set origin
        origin = np.random.normal(10, size=3)
        sitk_spacing = [1, 1, 4, 1]

        label_image = sitk.GetImageFromArray(labels.astype(np.uint8))
        label_image = sitk.Cast(label_image, sitk.sitkUInt8)
        label_image.SetSpacing(sitk_spacing)
        label_image.SetOrigin(origin)
        # write label file (make a sphere in the center with label one)
        sitk.WriteImage(label_image, str(label_file))

        # use sphere
        image_data = (
            np.repeat(np.expand_dims(labels, axis=3), repeats=cfg.num_channels, axis=3)
            * 128
        )
        # add noise
        image_data = image_data + np.abs(np.random.normal(size=shape, scale=128))
        image = sitk.GetImageFromArray(image_data)
        image.SetSpacing(sitk_spacing)
        image.SetOrigin(origin)
        # write image file
        sitk.WriteImage(image, str(image_file))

    return training_files


if __name__ == "__main__":
    # create test data if called
    create_test_files(Path("test_data"))
