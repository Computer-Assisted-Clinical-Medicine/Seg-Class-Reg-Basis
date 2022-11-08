"""Look for the available GPUs and display them"""
import os
import subprocess
import sys
from io import StringIO

import pandas as pd
import tensorflow as tf


def get_gpu_list() -> pd.DataFrame:
    """Get a list of GPUs in tf order with memory
    usage to display to the user for selection.

    Returns
    -------
    pd.DataFrame
        The GPU list
    """
    output = (
        subprocess.check_output(
            "nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,nounits",
            shell=True,
        )
        .decode(sys.stdout.encoding)
        .strip()
    )
    gpus_tf = tf.config.list_physical_devices("GPU")
    gpus_nvidia_smi = pd.read_csv(StringIO(output))
    gpu_devices = {
        tf.config.experimental.get_device_details(g)["device_name"]: g.name for g in gpus_tf
    }
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        to_drop = [n not in gpu_devices for n in gpus_nvidia_smi["name"]]
        gpus_nvidia_smi.drop(index=gpus_nvidia_smi.index[to_drop], inplace=True)
    cuda_num = {
        tf.config.experimental.get_device_details(g)["device_name"]: str(i)
        for i, g in enumerate(gpus_tf)
    }
    gpus_nvidia_smi["Number"] = gpus_nvidia_smi["name"].replace(cuda_num)
    gpus_nvidia_smi.sort_values("Number", inplace=True)
    gpus_nvidia_smi.set_index("Number", inplace=True)
    return gpus_nvidia_smi


if __name__ == "__main__":
    print(get_gpu_list())
