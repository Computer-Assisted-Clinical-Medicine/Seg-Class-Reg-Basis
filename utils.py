"""Different utilities to help with training and manage the experiments"""
import logging
import os
import stat
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf

# configure logger
logger = logging.getLogger(__name__)


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
    if "preferred_gpu" in os.environ:
        preferred_gpu = gpus.loc[int(os.environ["preferred_gpu"])]
    else:
        # get the GPU with the most free memory
        preferred_gpu = gpus.sort_values(" memory.free [MiB]").iloc[-1]
    free = preferred_gpu[" memory.free [MiB]"]
    if free > memory_limit:
        print(f"Using {preferred_gpu['name']}")
        logger.info("Using %s", preferred_gpu["name"])
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

    # remove unneeded dimensions for autoencoder
    if task == "autoencoder" and output.ndim == 4:
        if output.shape[3] == 1:
            output = output[:, :, :, 0]

    # turn the output into an image
    pred_img = sitk.GetImageFromArray(output)
    # cast to the right type
    if task in ("regression", "autoencoder") and output.ndim < 4:
        pred_img = sitk.Cast(pred_img, sitk.sitkFloat32)
    elif task in ("regression", "autoencoder") and output.ndim == 4:
        pred_img = sitk.Cast(pred_img, sitk.sitkVectorFloat32)
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


def export_npz(
    output: List[np.ndarray], tasks: List[str], task_names: List[str], file_path: Path
):
    """Export the output of the network as npz file

    Parameters
    ----------
    output : List[np.ndarray]
        The output of the network
    tasks : List[str]
        The list of tasks to performed
    task_names : List[str]
       The task names to be used as keys in the file
    file_path : Path
        The path where the file should be saved
    """
    assert len(task_names) == len(output)
    output_dict = {}
    for out, tsk, name in zip(output, tasks, task_names):
        # for regression, just save the whole thing, it is not that big
        if tsk == "regression":
            output_dict[name] = out.astype(np.float16)
        # average over the output
        elif tsk == "classification":
            output_dict[name] = out.mean(axis=tuple(range(out.ndim - 1)))
            output_dict[name + "_std"] = out.std(axis=tuple(range(out.ndim - 1)))
            output_dict[name + "_median"] = np.median(out, axis=tuple(range(out.ndim - 1)))
        # for now, just use less data for segmentation
        elif tsk == "segmentation":
            output_dict[name] = out.astype(np.float16)
    np.savez_compressed(file_path, **output_dict)


### experiment running utils


def configure_logging(tf_logger: logging.Logger) -> logging.Logger:
    """Configure the logger, the handlers of the tf_logger are removed and both
    loggers are set to

    Parameters
    ----------
    tf_logger : logging.Logger
        The tensorflow logger, must be assigned before importing tensorflow

    Returns
    -------
    logging.Logger
        The base logger
    """
    # configure loggers
    logger_config = logging.getLogger()
    logger_config.setLevel(logging.DEBUG)

    tf_logger.setLevel(logging.DEBUG)
    # there is too much output otherwise
    for handler in tf_logger.handlers:
        tf_logger.removeHandler(handler)
    return logger_config


def generate_res_path(version: str, external: bool, postprocessed: bool, task: str):
    """For a given path, generate the relative path to the result file"""
    if postprocessed:
        version += "-postprocessed"
    if external:
        folder_name = f"results_external_testset_{version}_{task}"
    else:
        folder_name = f"results_test_{version}_{task}"
    res_path = Path(folder_name) / "evaluation-all-files.csv"
    return res_path


def export_hyperparameters(experiments, experiment_dir):
    """
    Export a summary of the experiments and compare the hyperparameters of all experiments
    and collect the ones that were changed.
    """
    # export the hyperparameters
    experiments_file = experiment_dir / "experiments.csv"
    hyperparameter_changed_file = experiment_dir / "hyperparameters_changed.csv"
    # collect all results
    hparams = []
    for exp in experiments:
        # and parameters
        hparams.append(
            {
                **exp.hyper_parameters["network_parameters"],
                **exp.hyper_parameters["train_parameters"],
                "normalizing_method": exp.hyper_parameters["preprocessing_parameters"][
                    "normalizing_method"
                ],
                "loss": exp.hyper_parameters["loss"],
                "architecture": exp.hyper_parameters["architecture"].__name__,
                "dimensions": exp.hyper_parameters["dimensions"],
                "path": str(exp.output_path_rel),
                "exp_group_name": str(exp.output_path_rel.parent.name),
            }
        )

    # convert to dataframes
    hparams = pd.DataFrame(hparams)
    # find changed parameters
    changed_params = []
    # drop the results file when analyzing the changed hyperparameters
    for col in hparams:
        if hparams[col].astype(str).unique().size > 1:
            changed_params.append(col)
    # have at least one changed parameters (for the plots)
    if len(changed_params) == 0:
        changed_params = ["architecture"]
    hparams_changed = hparams[changed_params].copy()
    # if n_filters, use the first
    if "n_filters" in hparams_changed:
        hparams_changed.loc[:, "n_filters"] = (
            hparams_changed["n_filters"].dropna().apply(lambda x: x[0])
        )
    if "normalizing_method" in hparams_changed:
        n_name = hparams_changed["normalizing_method"].apply(lambda x: x.name)
        hparams_changed.loc[:, "normalizing_method"] = n_name
    # ignore the batch size (it correlates with the dimension)
    if "batch_size" in hparams_changed:
        hparams_changed.drop(columns="batch_size", inplace=True)
    # ignore do_bias (it is set the opposite to batch_norm)
    if "do_bias" in hparams_changed and "do_batch_normalization" in hparams_changed:
        hparams_changed.drop(columns="do_bias", inplace=True)
    # drop column specifying the files
    if "path" in hparams_changed:
        hparams_changed.drop(columns="path", inplace=True)
    # drop columns only related to architecture
    if "architecture" in hparams_changed:
        arch_groups = hparams_changed.astype(str).groupby("architecture")
        # there should be at least one other column
        if arch_groups.ngroups > 1 and hparams_changed.shape[1] > 1:
            arch_params = arch_groups.nunique(dropna=False)
            for col in arch_params:
                if np.all(arch_params[col] == 1):
                    hparams_changed.drop(columns=col, inplace=True)

    hparams.to_csv(experiments_file, sep=";")
    hparams.to_json(experiments_file.with_suffix(".json"))
    hparams_changed.to_csv(hyperparameter_changed_file, sep=";")
    hparams_changed.to_json(hyperparameter_changed_file.with_suffix(".json"))


def gather_results(
    experiment_dir: Path,
    task: str,
    external=False,
    postprocessed=False,
    combined=False,
    version="best",
) -> pd.DataFrame:
    """Collect all result files from all experiments. Only experiments that are
    already finished will be included in the analysis.

    Parameters
    ----------
    experiment_dir : Pathlike
        The path where the experiments are located
    task : str
        The task to analyze, choices are segmentation, classification and regression
    external : bool, optional
        If the external testset should be evaluated, by default False
    postprocessed : bool, optional
        If the data from the postprocessed should be evaluated, by default False
    combined : bool, optional
        If there is a combined model, which should be analyzed, by default True
    version : str, optional
        Which version of the model should be used, by default best

    Returns
    -------
    pd.DataFrame
        The results with all metrics for all files
    """
    experiments_file = experiment_dir / "experiments.json"

    if external:
        file_field = "results_file_external_testset"
    else:
        file_field = "results_file"

    if postprocessed:
        file_field += "_postprocessed"

    res_path = generate_res_path(version, external, postprocessed, task)

    hparams = pd.read_json(experiments_file)
    # type is incorrectly detected
    # pylint: disable=no-member

    # add combined model if present
    if combined:
        c_path = Path(hparams.iloc[0]["path"]).parent / "combined_models"
        loc = hparams.shape[0]
        hparams.loc[loc] = "Combined"
        hparams.loc[loc, "path"] = c_path

    # ignore some fields
    ignore = ["tasks", "label_shapes", "path"]

    results_all_list = []
    for _, row in hparams.iterrows():
        results_file = experiment_dir.parent / row["path"] / res_path
        if results_file.exists():
            results = pd.read_csv(results_file, sep=";")
            # set the model
            results["name"] = Path(row["path"]).name
            # set the other parameters
            for name, val in row.iteritems():
                if name in ignore:
                    continue
                results[name] = [val] * results.shape[0]
            # save results
            results_all_list.append(results)
        else:
            name = Path(results_file).parent.parent.name
            print(f"Could not find the evaluation file for {name}")

    if len(results_all_list) == 0:
        print("No files found")
        return None
    else:
        results_all = pd.concat(results_all_list)

    complete_percent = int(np.round(len(results_all_list) / hparams.shape[0] * 100))
    print(f"{complete_percent:3d} % of experiments completed.")

    # drop first column (which is just the old index)
    results_all.drop(results_all.columns[0], axis="columns", inplace=True)
    results_all["fold"] = pd.Categorical(results_all["fold"])
    results_all["name"] = pd.Categorical(results_all["name"])
    results_all["version"] = version
    results_all.index = pd.RangeIndex(results_all.shape[0])
    results_all.sort_values("File Number", inplace=True)
    return results_all


def gather_training_data(
    experiment_dir: Path,
) -> pd.DataFrame:
    """Collect all training data from all experiments.

    Parameters
    ----------
    experiment_dir : Pathlike
        The path where the experiments are located

    Returns
    -------
    pd.DataFrame
        The results with all metrics for all files
    """
    experiments_file = experiment_dir / "experiments.json"

    hparams = pd.read_json(experiments_file)
    # type is incorrectly detected
    # pylint: disable=no-member

    # ignore some fields
    ignore = ["tasks", "label_shapes", "path"]

    training_data_all_list = []
    for _, row in hparams.iterrows():
        model_dir = experiment_dir.parent / row["path"]
        for fold_dir in model_dir.glob("fold-*"):
            training_data_file = fold_dir / "training.csv"
            if training_data_file.exists():
                training_data = pd.read_csv(training_data_file, sep=";")
                # set the model
                training_data["name"] = Path(row["path"]).name
                training_data["fold"] = fold_dir.name
                # set the other parameters
                for name, val in row.iteritems():
                    if name in ignore:
                        continue
                    training_data[name] = [val] * training_data.shape[0]
                # save training_data
                training_data_all_list.append(training_data)

    if len(training_data_all_list) == 0:
        print("No files found")
        return None
    else:
        training_data_all = pd.concat(training_data_all_list)

    # drop first column (which is just the old index)
    training_data_all["fold"] = pd.Categorical(training_data_all["fold"])
    training_data_all.index = pd.RangeIndex(training_data_all.shape[0])
    return training_data_all


def export_slurm_job(
    filename,
    command,
    job_name=None,
    workingdir=None,
    venv_dir="venv",
    job_type="CPU",
    cpus=1,
    hours=0,
    minutes=30,
    log_dir=None,
    log_file=None,
    error_file=None,
    array_job=False,
    array_range="0-4",
    singleton=False,
    variables=None,
):
    """Generates a slurm file to run jobs on the cluster

    Parameters
    ----------
    filename : Path or str
        Where the slurm file should be saved
    command : str
        The command to run (can also be multiple commands separated by line breaks)
    job_name : str, optional
        The name displayed in squeue and used for log_name, by default None
    workingdir : str, optional
        The directory in Segmentation_Experiment, if None, basedir is used, by default None
    venv_dir : str, optional
        The directory of the virtual environment, by default venv
    job_type : str, optional
        type of job, CPU, GPU or GPU_no_K80, by default 'CPU'
    cpus : int, optional
        number of CPUs, by default 1
    hours : int, optional
        Time the job should run in hours, by default 0
    minutes : int, optional
        Time the job should run in minutes, by default 30
    log_dir : str, optional
        dir where the logs should be saved if None logs/job_name/, by default None
    log_file : str, optional
        name of the log file, if None job_name_job_id_log.txt, by default None
    error_file : str, optional
        name of the errors file, if None job_name_job_id_log_errors.txt, by default None
    array_job : bool, optional
        If set to true, array_range should be set, by default False
    array_range : str, optional
        array_range as str (comma separated or start-stop (ends included)), by default '0-4'
    singleton : bool, optional
        if only one job with that name and user should be running, by default False
    variables : dict, optional
        environmental variables to write {name : value} $EXPDIR can be used, by default {}
    """

    if variables is None:
        variables = {}

    # this new node dos not work
    exclude_nodes = ["h08c0301", "h08c0401", "h08c0501"]
    if job_type == "GPU_no_K80":
        exclude_nodes += [
            "h05c0101",
            "h05c0201",
            "h05c0301",
            "h05c0401",
            "h05c0501",
            "h06c0301",
            "h05c0601",
            "h05c0701",
            "h05c0801",
            "h05c0901",
            "h06c0101",
            "h06c0201",
            "h06c0401",
            "h06c0501",
            "h06c0601",
            "h06c0701",
            "h06c0801",
            "h06c0901",
        ]

    if job_type == "CPU":
        assert hours == 0
        assert minutes <= 30
    else:
        assert minutes < 60
        assert hours <= 48

    if log_dir is None:
        log_dir = Path("logs/{job_name}/")
    else:
        log_dir = Path(log_dir)

    if log_file is None:
        if array_job:
            log_file = log_dir / f"{job_name}_%a_%A_log.txt"
        else:
            log_file = log_dir / f"{job_name}_%j_log.txt"
    else:
        log_file = log_dir / log_file

    if error_file is None:
        if array_job:
            error_file = log_dir / f"{job_name}_%a_%A_errors.txt"
        else:
            error_file = log_dir / f"{job_name}_%j_errors.txt"
    else:
        error_file = log_dir / error_file

    filename = Path(filename)

    slurm_file = "#!/bin/bash\n\n"
    if job_name is not None:
        slurm_file += f"#SBATCH --job-name={job_name}\n"

    slurm_file += f"#SBATCH --cpus-per-task={cpus}\n"
    slurm_file += "#SBATCH --ntasks-per-node=1\n"
    slurm_file += f"#SBATCH --time={hours:02d}:{minutes:02d}:00\n"
    slurm_file += "#SBATCH --mem=32gb\n"

    if job_type in ("GPU", "GPU_no_K80"):
        slurm_file += "\n#SBATCH --partition=gpu-single\n"
        slurm_file += "#SBATCH --gres=gpu:1\n"

    if len(exclude_nodes) > 0:
        slurm_file += "#SBATCH --exclude=" + ",".join(exclude_nodes) + "\n"

    if array_job:
        slurm_file += f"\n#SBATCH --array={array_range}\n"

    # add logging
    slurm_file += f"\n#SBATCH --output={str(log_file)}\n"
    slurm_file += f"#SBATCH --error={str(error_file)}\n"

    if singleton:
        slurm_file += "\n#SBATCH --dependency=singleton\n"

    # define workdir, add diagnostic info
    slurm_file += """
echo "Set Workdir"
WSDIR=/gpfs/bwfor/work/ws/hd_mo173-myws
echo $WSDIR
EXPDIR=$WSDIR\n"""

    # print task ID depending on type
    if array_job:
        slurm_file += '\necho "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID\n'
    else:
        slurm_file += '\necho "My SLURM_JOB_ID: " $SLURM_JOB_ID\n'

    slurm_file += """\necho "job started on Node: $HOSTNAME"

echo "Load modules"

module load devel/python_intel/3.7
"""

    # add environmental variables
    if len(variables) > 0:
        slurm_file += "\n"
    for key, val in variables.items():
        slurm_file += f'export {key}="{val}"\n'

    if "GPU" in job_type:
        slurm_file += """module load devel/cuda/10.1
module load lib/cudnn/7.6.5-cuda-10.1

echo "Get GPU info"
nvidia-smi
"""

    slurm_file += '\necho "Go to workingdir"\n'
    if workingdir is None:
        slurm_file += "cd $EXPDIR/nnUNet\n"
    else:
        slurm_file += f"cd {Path(workingdir).resolve()}\n"

    # activate virtual environment
    slurm_file += '\necho "Activate virtual environment"\n'
    slurm_file += f"source {Path(venv_dir).resolve()}/bin/activate\n"

    # run the real command
    slurm_file += '\necho "Start calculation"\n\n'
    slurm_file += command
    slurm_file += '\n\necho "Finished"'

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)
    # write to file
    with open(filename, "w+", encoding="utf8") as f:
        f.write(slurm_file)


def export_batch_file(filename, commands):
    """Exports a list of commands (one per line) as batch script

    Parameters
    ----------
    filename : str or Path
        The new file
    commands : [str]
        List of commands (as strings)
    """

    filename = Path(filename)

    batch_file = "#!/bin/bash"

    for com in commands:
        batch_file += f"\n\n{com}"

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)
    # write to file
    with open(filename, "w+", encoding="utf8") as f:
        f.write(batch_file)

    # set permission
    os.chmod(filename, stat.S_IRWXU)


def export_powershell_scripts(script_dir: Path, experiments: list):
    """Export power shell script to start the different folds and to start tensorboard.

    Parameters
    ----------
    script_dir : Path
        The directory where the scripts should be placed
    experiments : List[Experiment]
        The experiments to export
    """
    # set the environment (might be changed for each machine)
    first_exp = experiments[0]
    experiment_dir = first_exp.experiment_dir
    assert isinstance(experiment_dir, Path)
    data_dir = Path(os.environ["data_dir"])
    ps_script_set_env = experiment_dir / "set_env.ps1"
    python_script_dir = Path(sys.argv[0]).resolve().parent
    command = f'$env:script_dir="{python_script_dir}"\n'
    command += "$env:script_dir=$env:script_dir -replace ' ', '` '\n"
    command += f'$env:data_dir="{data_dir}"\n'
    command += f'$env:experiment_dir="{experiment_dir}"\n'
    if "preferred_gpu" in os.environ:
        pref_gpu = os.environ["preferred_gpu"]
        command += f"$env:preferred_gpu={pref_gpu}\n"

    # create env file
    if not ps_script_set_env.exists():
        with open(ps_script_set_env, "w+", encoding="utf8") as powershell_file_tb:
            powershell_file_tb.write(command)

    ps_script = script_dir / "start.ps1"
    eval_script = script_dir / "start_eval.ps1"
    ps_script_tb = script_dir / "start_tensorboard.ps1"

    # make a powershell command, add env
    if script_dir.resolve() == experiment_dir.resolve():
        command = '$set_env=".\\set_env.ps1"\n'
    else:
        command = "$script_parent = (get-item $PSScriptRoot ).parent.FullName\n"
        command += '$set_env="${script_parent}\\set_env.ps1"\n'
    command += "$set_env=$set_env -replace ' ', '` '\n"
    command += "Invoke-Expression ${set_env}\n"
    command += 'Write-Output "Data dir: $env:data_dir"\n'
    command += 'Write-Output "Experiment dir: $env:experiment_dir"\n'
    command += 'Write-Output "Script dir: $env:script_dir"\n'

    # activate
    command += 'Write-Output "Activate Virtual Environment"\n'
    command += '$activate=${env:script_dir} + "\\venv\\Scripts\\activate.ps1"\n'
    command += "Invoke-Expression ${activate}\n"

    # tensorboard command (up to here, it is the same)
    command_tb = command
    command_tb += "$start='tensorboard --logdir=\"' + "
    if script_dir.resolve() == experiment_dir.resolve():
        rel_dir = ""
    else:
        rel_dir = str(script_dir.relative_to(experiment_dir))
    command_tb += f"${{env:experiment_dir}} + '\\{rel_dir}\"'\n"
    command_tb += "Write-Output $start\n"
    command_tb += "Invoke-Expression ${start}\n"

    # add the experiments
    command += '\n\n\n$script_run=${env:script_dir} + "\\run_single_experiment.py"\n'
    command_eval = (
        command + '$script_eval=${env:script_dir} + "\\evaluate_single_experiment.py"\n'
    )
    for exp in experiments:
        command_path = f'$output_path=${{env:experiment_dir}} + "\\{exp.output_path_rel}"\n'
        command += command_path
        command_eval += command_path
        for fold_num in range(exp.folds):
            fold_task_name = f"{exp.output_path_rel.parent.name}-{exp.name} Fold {fold_num}"
            command += f'Write-Output "starting with {fold_task_name}"\n'
            command_eval += f'Write-Output "starting with {fold_task_name}"\n'
            command += f'$command="python " + ${{script_run}} + " -f {fold_num} -e " + \'${{output_path}}\'\n'
            command += "Invoke-Expression ${command}\n\n"
            command_eval += f'$command="python " + ${{script_eval}} + " -f {fold_num} -e " + \'${{output_path}}\'\n'
            command_eval += "Invoke-Expression ${command}\n\n"

    with open(ps_script, "w+", encoding="utf8") as powershell_file:
        powershell_file.write(command)

    with open(eval_script, "w+", encoding="utf8") as powershell_file:
        powershell_file.write(command_eval)

    # create tensorboard file
    with open(ps_script_tb, "w+", encoding="utf8") as powershell_file_tb:
        powershell_file_tb.write(command_tb)

    print(f"To run the training, execute {ps_script}")
    print(f"To run tensorboard, execute {ps_script_tb}")


def export_experiments_run_files(script_dir: Path, experiments: list):
    """Export the files to run the experiments. These are first the hyperparameter
    comparison files and then depending on the environment (Windows or Linux cluster),
    either bash script to submit slurm jobs or powershell scripts to start the
    experiments are written.

    Parameters
    ----------
    script_dir : Path
        The directory where the scripts should be placed
    experiments : List[Experiment]
        The experiments to export
    """

    # export all hyperparameters
    export_hyperparameters(experiments, script_dir)

    # if on cluster, export slurm files
    if "CLUSTER" in os.environ:
        slurm_files = []
        working_dir = Path("").resolve()
        if not working_dir.exists():
            working_dir.mkdir()
        for exp in experiments:
            slurm_files.append(exp.export_slurm_file(working_dir))

        start_all_batch = script_dir / "start_all_jobs.sh"
        export_batch_file(
            filename=start_all_batch,
            commands=[f"sbatch {f}" for f in slurm_files],
        )

        # and create some needed directories (without their log dirs, jobs don't start)
        plot_dir_slurm = working_dir / "plots" / "slurm"
        if not plot_dir_slurm.exists():
            plot_dir_slurm.mkdir(parents=True)
        combined_dir_slurm = working_dir / "combined_models" / "slurm"
        if not combined_dir_slurm.exists():
            combined_dir_slurm.mkdir(parents=True)
        print(f"To start the training, execute {start_all_batch}")
    # if on local computer, export powershell start file
    else:
        export_powershell_scripts(script_dir, experiments)
