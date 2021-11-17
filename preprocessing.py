"""
Preprocess the input images
"""
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from . import config as cfg

# configure logger
logger = logging.getLogger(__name__)


class NoLabelsInOverlap(Exception):
    def __init__(self):
        super().__init__("Labels were not in the overlapping region")


def load_image(path: os.PathLike):
    return sitk.ReadImage(str(path))


def save_image(image: sitk.Image, path: os.PathLike):
    sitk.WriteImage(image, str(path))


def combine_images(
    images: List[sitk.Image],
    labels: Optional[sitk.Image],
    resample=True,
    target_spacing=(1, 1, 3),
) -> Tuple[sitk.Image, Optional[sitk.Image]]:
    """Combine images by calculating the best overlap and the combining the images
    into one composed image. The images will also be resampled if specified.
    The first image will be used as reference image for the direction.

    Parameters
    ----------
    images : List[sitk.Image]
        List of images
    labels : Optional[sitk.Image]
        Labels, which will be resampled to the same coordinates as the images
    resample : bool, optional
        if resampling should be done, by default True
    target_spacing : tuple, optional
        The spacing for the resampling, by default (1, 1, 3)

    Returns
    -------
    Tuple[sitk.Image, Optional[sitk.Image]]
        The resulting images, if no labels were given, the second image will be None.
    """
    overlap = get_overlap(images)
    # take the first image as reference image
    reference_image: sitk.Image = images[0][overlap]
    if labels is not None:
        num_labels = sitk.GetArrayFromImage(labels).astype(bool).sum()
        if num_labels == 0:
            logger.warning("No labels found in the image.")
    # resample reference image to the new spacing if wanted
    if resample:
        assert target_spacing is not None
        # calculate spacing
        orig_size = np.array(reference_image.GetSize())
        orig_spacing = np.array(reference_image.GetSpacing())

        # set new sizes
        physical_size = orig_size * orig_spacing
        new_size = (physical_size / target_spacing).astype(int)
        new_physical_size = new_size * target_spacing

        # and origin
        orig_origin = np.array(reference_image.GetOrigin())
        size_diff = physical_size - new_physical_size
        direction = np.array(reference_image.GetDirection()).reshape((3, 3)).T
        shift = np.dot((size_diff / 2), direction)
        new_origin = orig_origin + shift

        # see how much the number of labels is reduced
        if labels is not None:
            orig_label_spacing = np.array(labels.GetSpacing())
            # see how much the number of voxels in the same are is reduced
            reduction_factor = np.prod(target_spacing / orig_label_spacing)

        resample_method = sitk.ResampleImageFilter()
        resample_method.SetDefaultPixelValue(0)
        resample_method.SetInterpolator(sitk.sitkBSplineResampler)
        resample_method.SetOutputDirection(reference_image.GetDirection())
        # for some reason, there is an error otherwise
        resample_method.SetSize([int(n) for n in new_size])
        resample_method.SetOutputOrigin(list(new_origin))
        resample_method.SetOutputPixelType(sitk.sitkFloat32)
        resample_method.SetOutputSpacing(target_spacing)
        reference_image_resized = resample_method.Execute(reference_image)

        if labels is not None:
            # also resample labels, but change interpolator
            resample_method.SetInterpolator(sitk.sitkNearestNeighbor)
            resample_method.SetOutputPixelType(sitk.sitkUInt8)
            labels_resampled = resample_method.Execute(labels)

    # sample all images to the reference image
    resample_method = sitk.ResampleImageFilter()
    resample_method.SetReferenceImage(reference_image_resized)
    resample_method.SetOutputPixelType(sitk.sitkFloat32)
    resample_method.SetInterpolator(sitk.sitkBSplineResampler)

    images_resampled = [reference_image_resized] + [
        resample_method.Execute(img) for img in images[1:]
    ]

    image_combine = sitk.Compose(images_resampled)

    # check labels
    if labels_resampled is not None:
        num_labels_res = sitk.GetArrayFromImage(labels_resampled).astype(bool).sum()
        if num_labels != 0 and num_labels_res < 10:
            raise NoLabelsInOverlap()
        # account for the differences in the spacing when seeing if labels are missing
        if num_labels_res < num_labels / reduction_factor * 0.85:
            logger.warning("Less labelled voxels in the resampled image.")

    return image_combine, labels_resampled


def generate_mask(image: sitk.Image, lower_quantile=0.25) -> sitk.Image:
    """Generate a binary mask by only considering pixels between the lower quantile
    and the maximum value

    Parameters
    ----------
    image : sitk.Image
        The image
    lower_quantile : float, optional
        The quantile to use for the low value, by default 0.25

    Returns
    -------
    sitk.Image
        The mask
    """
    image_np = sitk.GetArrayFromImage(image)
    upper = np.max(image_np)
    lower = np.quantile(image_np, lower_quantile)
    mask_image = sitk.BinaryThreshold(
        image1=image,
        lowerThreshold=lower + 1e-9,  # add small value to exclude the limit
        upperThreshold=upper + 1e-9,  # for the higher values, include the limit
    )
    mask_closed = sitk.BinaryClosingByReconstruction(
        image1=mask_image, kernelRadius=[5, 5, 5]
    )
    mask_opened = sitk.BinaryOpeningByReconstruction(
        image1=mask_closed, kernelRadius=[2, 2, 2]
    )
    return mask_opened


def get_overlap(images: List[sitk.Image]) -> list:
    """Calculate the area of overlap between all three images

    Parameters
    ----------
    images : List[sitk.Image]
        The images, the coordinate system of the first image will be used

    Returns
    -------
    list
        List with start and stop of valid indices for all images
    """
    images_mask = [generate_mask(img) for img in images]
    # resample to the first image
    masks_resampled = [images_mask[0]]
    for img in images_mask[1:]:
        img_resampled = sitk.Resample(
            image1=img,
            referenceImage=images_mask[0],
            transform=sitk.Euler3DTransform(),
            interpolator=sitk.sitkNearestNeighbor,
        )
        masks_resampled.append(img_resampled)
    # turn into numpy arrays
    mask_np = np.array([sitk.GetArrayFromImage(img) for img in masks_resampled]).astype(
        bool
    )
    all_mask = np.all(mask_np, axis=0)
    valid_indices = []
    # use opposite order as SimpleITK does
    axes_names = ["z", "y", "x"]
    for i in range(2, -1, -1):
        avg_axes = tuple(set(range(3)) - set([i]))
        dir_name = axes_names[i]
        valid = all_mask.mean(axis=avg_axes) > 0.2
        # get the first good value (start will include it)
        start = np.argmax(valid)
        # get the last good value (the end will exclude the limit, but size is off by 1)
        end = valid.size - np.argmax(np.flip(valid))
        valid_indices.append(slice(start, end))
        frac_removed = (valid.size - (end - start)) / valid.size
        if frac_removed > 0.3:
            logger.warning(
                "More than 30 %% (%i %%) removed in direction %s.",
                int(frac_removed * 100),
                dir_name,
            )
        elif frac_removed > 0.5:
            raise Exception(
                f"More that 50% ({frac_removed} %) removed in direction {dir_name}."
            )

    return valid_indices


def preprocess_dataset(
    data_set,
    num_channels,
    base_dir,
    preprocessed_dir,
    train_dataset,
    preprocessing_parameters,
):
    """Preprocess the images by applying the normalization and then combining
    them into one image.
    """
    if not preprocessed_dir.exists():
        preprocessed_dir.mkdir(parents=True)
    # load normalization
    normalization_method = preprocessing_parameters["normalizing_method"]
    normalization_class = normalization_method.get_class()
    # make lists to train the normalization
    norm_train_set = [[] for n in range(num_channels)]
    for name in train_dataset:
        images = data_set[name]["images"]
        assert len(images) == num_channels, "Number of modalities inconsistent"
        for num, img in enumerate(images):
            norm_train_set[num].append(img)
    # train the normalization
    normalizations = []
    for num in range(num_channels):
        norm_file = base_dir / preprocessed_dir / f"normalization_mod{num}.yaml"
        if not norm_file.parent.exists():
            norm_file.parent.mkdir(parents=True)
        if norm_file.exists():
            norm = normalization_class.from_file(norm_file)
            # make sure the parameters are correct
            parameters_from_file = norm.get_parameters()
            for key, value in preprocessing_parameters["normalization_parameters"].items():
                if not parameters_from_file[key] == value:
                    raise ValueError(
                        "Normalization of preprocessed images has different parameters."
                    )
        else:
            norm = normalization_class(
                **preprocessing_parameters["normalization_parameters"]
            )
            images_gen = (load_image(img) for img in norm_train_set[num])
            norm.train_normalization(images_gen)
            # and save it
            norm.to_file(norm_file)
        normalizations.append(norm)

    # remember preprocessed images
    preprocessed_dict = {}

    # resample and apply normalization
    for name, data in tqdm(data_set.items(), unit="image"):
        # define paths
        image_paths = data["images"]
        image_rel_path = preprocessed_dir / str(
            cfg.sample_file_name_prefix + name + cfg.file_suffix
        )
        image_processed_path = base_dir / image_rel_path
        labels_exist = True
        if "labels" in data:
            labels_path = data["labels"]
            label_rel_path = preprocessed_dir / str(
                cfg.label_file_name_prefix + name + cfg.file_suffix
            )
            labels_processed_path = base_dir / label_rel_path
        else:
            labels_exist = False

        # preprocess images
        preprocess_image(
            image_paths=image_paths,
            image_processed_path=image_processed_path,
            labels_path=labels_path,
            labels_processed_path=labels_processed_path,
            normalizations=normalizations,
            preprocessing_parameters=preprocessing_parameters,
        )

        if image_processed_path.exists():
            preprocessed_dict[str(name)] = {}
            preprocessed_dict[str(name)]["image"] = image_rel_path
        if labels_exist:
            if labels_processed_path.exists():
                preprocessed_dict[str(name)]["labels"] = label_rel_path

    logger.info("Preprocessing finished.")

    return preprocessed_dict


def preprocess_image(
    image_paths: List[Path],
    image_processed_path: Path,
    labels_path: Optional[Path],
    labels_processed_path: Optional[Path],
    normalizations: List[Callable],
    preprocessing_parameters: Dict,
):
    """Preprocess a single image, it will be normalized and then all images
    will be combined into one

    Parameters
    ----------
    name : str
        The name (used in the filename)
    image_paths : List[Path]
        The path of the image
    labels_path : Optional[Path]
        The path of the labels (or None, then no labels are exported)
    normalizations : List
        A list of the normalizations used to process each image, should be the
        same length as the images
    """
    # preprocess if it does not exist yet
    if not image_processed_path.exists():
        # load and normalize images
        images = [load_image(img) for img in image_paths]
        images_normalized = [norm(img) for img, norm in zip(images, normalizations)]
        if labels_path is not None:
            labels = load_image(labels_path)
        else:
            labels = None

        logger.info("Processing image %s", image_processed_path.name)
        try:
            res_image, res_labels = combine_images(
                images=images_normalized,
                labels=labels,
                resample=preprocessing_parameters["resample"],
                target_spacing=preprocessing_parameters["target_spacing"],
            )
        except NoLabelsInOverlap as exc:
            print("Labels were not in the overlapping image portion.")
            logger.exception(exc)
        else:
            save_image(res_image, image_processed_path)
            if labels is not None:
                assert (
                    labels_processed_path is not None
                ), "if there are labels, also provide a path."
                save_image(res_labels, labels_processed_path)
