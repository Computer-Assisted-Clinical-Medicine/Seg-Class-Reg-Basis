"""Different utilities to help with training"""
import os
import subprocess
import sys
from io import StringIO

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf


def get_gpu(memory_limit=4000) -> str:
    """Get the name of the GPU with the most free memory as required by tensorflow

    Parameters
    ----------
    memory_limit : int, optional
        The minimum free memory in MB, by default 4000

    Returns
    -------
    str
        The GPU with the most free memory

    Raises
    ------
    SystemError
        If not free GPU is available
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    output = (
        subprocess.check_output(
            "nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,nounits",
            shell=True,
        )
        .decode(sys.stdout.encoding)
        .strip()
    )
    tf_gpus = [device.name for device in tf.config.list_physical_devices("GPU")]
    gpus = pd.read_csv(StringIO(output))
    gpus["tf_name"] = tf_gpus
    # get the GPU with the most free memory
    preferred_gpu = gpus.sort_values(" memory.free [MiB]").iloc[-1]
    free = preferred_gpu[" memory.free [MiB]"]
    if free > memory_limit:
        print(f"Using {preferred_gpu['name']}")
        return preferred_gpu.tf_name.partition("physical_device:")[2]
    else:
        raise SystemError("No free GPU available")


def output_to_image(
    output: np.ndarray,
    task: str,
    processed_image: sitk.Image,
    original_image: sitk.Image,
) -> sitk.Image:
    """Convert the network output to an image. For classification and segmentation,
    argmax is applied first to the last dimension. Then, the output is converted to
    an image with the same physical dimensions as the processed image. For segmentation,
    it is then also resampled to the original image.

    Parameters
    ----------
    output : np.ndarray
        The output to process
    task : str
        The name of the task, it should be "segmentation", "classification" or "regression".
    processed_image : sitk.Image
        The processed image used for the prediction
    original_image : sitk.Image
        The original image, only needed for segmentation

    Returns
    -------
    sitk.Image
        The resulting Image
    """
    if output.ndim > 4:
        raise ValueError("Result should have at most 4 dimensions")
    # make sure that the output has the right number of dimensions
    if task == "segmentation" and output.ndim != 4:
        raise ValueError("For segmentation, a 4D Result is expected.")
    # for classification, add dimensions until there are 4
    if task == "classification":
        if output.ndim < 4:
            output = np.expand_dims(
                output, axis=tuple(2 - i for i in range(4 - output.ndim))
            )
    # a regression task should have just 3 dimensions
    elif task == "regression":
        if output.ndim < 3:
            output = np.expand_dims(
                output, axis=tuple(i + 1 for i in range(3 - output.ndim))
            )
        elif output.ndim == 4:
            raise ValueError("For regression, there should only be 3 dimensions.")

    # do the prediction for classification tasks
    if task in ("segmentation", "classification"):
        output = np.argmax(output, axis=-1)

    # turn the output into an image
    pred_img = sitk.GetImageFromArray(output)
    # cast to the right type
    if task == "regression":
        pred_img = sitk.Cast(pred_img, sitk.sitkFloat32)
    else:
        pred_img = sitk.Cast(pred_img, sitk.sitkUInt8)
    image_size = np.array(processed_image.GetSize()[:3])  # image could be 4D
    zoom_factor = image_size / pred_img.GetSize()

    # set the image information, the extent should be constant
    pred_img.SetDirection(processed_image.GetDirection())
    pred_img.SetSpacing(processed_image.GetSpacing() * zoom_factor)
    # in each direction, the origin is shifted by half the zoom factor, but there
    # is a shift by 1, because the origin is at the center of the first voxel
    new_origin_idx = (zoom_factor - 1) / 2
    pred_img.SetOrigin(
        processed_image.TransformContinuousIndexToPhysicalPoint(new_origin_idx)
    )

    if task == "segmentation":
        pred_img = sitk.Resample(
            image1=pred_img,
            referenceImage=original_image,
            interpolator=sitk.sitkNearestNeighbor,
            outputPixelType=sitk.sitkUInt8,
        )

    return pred_img


def compress_output(output: np.ndarray, task: str) -> np.ndarray:
    """Compress the output. For regression, it is converted to float16, probabilites are converted to 1 / int8

    Parameters
    ----------
    output : np.ndarray
        The output of the network
    task : str
        The task that was performed

    Returns
    -------
    np.ndarray
        The resulting array
    """
    # for regression, use float16
    if task == "regression":
        output_red = output.astype(np.float16)
    # for classification, just save 8 bit
    else:
        output_red = (1 / output).astype(np.uint8)
    return output_red
