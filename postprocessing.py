"""Different methods for postprocessing of labelled images
"""
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


def keep_big_structures(input_image: sitk.Image, target_image, n_keep=1):
    """In the labelled image, keep the n_keep largest structures (by volume)

    Parameters
    ----------
    input_image : sitk.Image
        The label image
    target_image : pathlike
        The place where the new image should be saved
    n_keep : int, optional
        The number of structures to keep, by default 1
    """

    input_sitk = sitk.ReadImage(str(input_image))
    input_numpy = sitk.GetArrayFromImage(input_sitk)

    # label all objects (0 is background, rest is foreground)
    labels, n_labels = ndimage.label(input_numpy)
    n_voxels = np.bincount(labels.ravel())

    # convert to dataframe (and drop the background)
    n_pixels = pd.Series(n_voxels[1:], index=np.arange(n_labels) + 1, dtype=int)
    # select the labels to remove
    to_remove = n_pixels.sort_values(ascending=False)[n_keep:].index
    # create a mask
    mask = np.isin(labels, to_remove)
    # apply it
    input_numpy[mask] = 0

    # turn into image
    processed_image = sitk.GetImageFromArray(input_numpy)
    processed_image.CopyInformation(input_sitk)

    sitk.WriteImage(processed_image, str(target_image))
