"""
Collection of functions to evaluate and plot the results.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from . import metric as Metric

# configure logger
logger = logging.getLogger(__name__)
# disable the font manager logger
logging.getLogger("matplotlib.font_manager").disabled = True


def evaluate_segmentation_prediction(prediction_path: str, label_path: str) -> dict:
    """Evaluate different metrics for one image

    Parameters
    ----------
    result_metrics : dict
        The dict were the metrics will be written
    prediction_path : str
        The path of the predicted image
    label_path : str
        The path of the ground truth image

    Returns
    -------
    dict
        The dict with the resulting metrics
    """
    pred_img = sitk.ReadImage(prediction_path)
    result_metrics = {}
    result_metrics["Slices"] = pred_img.getSize()[2]

    # load label for evaluation
    label_img = sitk.ReadImage(label_path)

    # This is necessary as in some data sets this is incorrect.
    label_img.SetDirection(pred_img.GetDirection())
    label_img.SetOrigin(pred_img.GetOrigin())
    label_img.SetSpacing(pred_img.GetSpacing())

    # check types and if not equal, convert output to target
    if pred_img.GetPixelID() != label_img.GetPixelID():
        cast = sitk.CastImageFilter()
        cast.SetOutputPixelType(label_img.GetPixelID())
        pred_img = cast.Execute(pred_img)

    result_metrics["Volume (L)"] = Metric.get_ml_sitk(label_img)

    # check if all labels are background
    if np.all(sitk.GetArrayFromImage(pred_img) == 0):
        # if that is not the case, create warning and return metrics
        if not np.all(sitk.GetArrayFromImage(label_img) == 0):
            logger.warning("Only background labels found")
            # set values for results
            result_metrics["Volume (P)"] = 0
            result_metrics["Dice"] = 0
            result_metrics["False Negative"] = 0
            result_metrics["False Positive"] = 1
            result_metrics["Confusion Rate"] = 1
            result_metrics["Connectivity"] = 0
            result_metrics["Fragmentation"] = 1
            result_metrics["Hausdorff"] = np.NAN
            result_metrics["Mean Symmetric Surface Distance"] = np.NAN
            return result_metrics

    result_metrics["Volume (P)"] = Metric.get_ml_sitk(pred_img)

    orig_dice, orig_vs, orig_fn, orig_fp, orig_iou = Metric.overlap_measures_sitk(
        pred_img, label_img
    )
    result_metrics["Dice"] = orig_dice
    result_metrics["IoU"] = orig_iou
    # result_metrics['Volume Similarity'] = orig_vs/
    result_metrics["False Negative"] = orig_fn
    result_metrics["False Positive"] = orig_fp
    logger.info(
        "  Original Overlap Measures: %s %s %s %s", orig_dice, orig_vs, orig_fn, orig_fp
    )

    confusion_rate = Metric.confusion_rate_sitk(pred_img, label_img, 1, 0)
    result_metrics["Confusion Rate"] = confusion_rate
    logger.info("  Confusion Rate: %s", confusion_rate)

    connect = Metric.get_connectivity_sitk(pred_img)
    result_metrics["Connectivity"] = connect
    logger.info("  Connectivity: %s", connect)

    frag = Metric.get_fragmentation_sitk(pred_img)
    result_metrics["Fragmentation"] = frag
    logger.info("  Fragmentation: %s", frag)

    try:
        orig_hdd = Metric.hausdorff_metric_sitk(pred_img, label_img)
    except RuntimeError as err:
        logger.exception("Surface evaluation failed! Using infinity: %s", err)
        orig_hdd = np.NAN
    result_metrics["Hausdorff"] = orig_hdd
    logger.info("  Original Hausdorff Distance: %s", orig_hdd)

    try:
        (
            orig_mnssd,
            orig_mdssd,
            orig_stdssd,
            orig_maxssd,
        ) = Metric.symmetric_surface_measures_sitk(pred_img, label_img)
    except RuntimeError as err:
        logger.exception("Surface evaluation failed! Using infinity: %s", err)
        orig_mnssd = np.NAN
        orig_mdssd = np.NAN
        orig_stdssd = np.NAN
        orig_maxssd = np.NAN

    result_metrics["Mean Symmetric Surface Distance"] = orig_mnssd

    logger.info(
        "   Original Symmetric Surface Distance: %s (mean) %s (median) %s (STD) %s (max)",
        orig_mnssd,
        orig_mdssd,
        orig_stdssd,
        orig_maxssd,
    )

    return result_metrics


def evaluate_classification(
    predictions: np.ndarray, std: np.ndarray, ground_truth: int
) -> Dict[str, float]:
    """Evaluate a classification task

    Parameters
    ----------
    predictions : np.ndarray
        The predictions to evaluate as np.array
    ground_truth : int
        The ground truth as integer

    Returns
    -------
    Dict[str, float]
        The resulting metric with one entry per metric
    """
    metrics = {}
    class_prediction = np.argmax(predictions, axis=-1)
    metrics["accuracy"] = (class_prediction == ground_truth).mean()
    pred_mean = predictions.mean(axis=tuple(range(predictions.ndim - 1)))
    metrics["top_prediction"] = np.argmax(pred_mean)
    metrics["ground_truth"] = ground_truth
    metrics["std"] = std[class_prediction]
    return metrics


def evaluate_regression(predictions: np.ndarray, ground_truth: float) -> Dict[str, float]:
    """Evaluate a regression task

    Parameters
    ----------
    predictions : np.ndarray
        The predictions to evaluate as np.ndarray
    ground_truth : float
        The ground truth as float between 0 and 1

    Returns
    -------
    Dict[str, float]
        The resulting metric with one entry per metric
    """
    metrics = {}
    metrics["rmse"] = np.sqrt(np.mean(np.square((predictions - ground_truth))))
    metrics["mean_absolute_error"] = np.mean(np.abs((predictions - ground_truth)))
    metrics["median_absolute_error"] = np.median(np.abs((predictions - ground_truth)))
    metrics["largest_absolute_error"] = np.max(np.abs((predictions - ground_truth)))
    metrics["smallest_absolute_error"] = np.min(np.abs((predictions - ground_truth)))
    metrics["mean_prediction"] = predictions.mean()
    metrics["std"] = predictions.std()
    return metrics


def combine_evaluation_results_from_folds(results_path, eval_files: List):
    """Combine the results of the individual folds into one file and calculate
    the means and standard deviations in separate files

    Parameters
    ----------
    results_path : Pathlike
        The path where the results should be written
    eval_files : List
        A list of the eval files
    """
    if len(eval_files) == 0:
        logger.info("Eval files empty, nothing to combine")
        return

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    _, experiment = os.path.split(results_path)
    eval_mean_file_path = os.path.join(
        results_path, "evaluation-mean-" + experiment + ".csv"
    )
    eval_std_file_path = os.path.join(results_path, "evaluation-std-" + experiment + ".csv")
    all_statistics_path = os.path.join(results_path, "evaluation-all-files.csv")

    statistics_list = []
    for eval_f in eval_files:
        if not eval_f.exists():
            continue
        data = pd.read_csv(eval_f, sep=";")
        data["fold"] = eval_f.parent.name
        statistics_list.append(data)

    if len(statistics_list) > 0:
        # concatenate to one array
        statistics = pd.concat(statistics_list).sort_values(["File Number", "fold"])
        # write to file
        statistics.to_csv(all_statistics_path, sep=";")

        mean_statistics = statistics.groupby("fold").mean()
        mean_statistics.to_csv(eval_mean_file_path, sep=";")

        std_statistics = statistics.groupby("fold").std()
        std_statistics.to_csv(eval_std_file_path, sep=";")


def make_boxplot_graphic(results_path: Path, result_file: Path):
    """Make a boxplot of the resulting metrics

    Parameters
    ----------
    results_path : Path
        Plots where the plots should be saved (a plot directory is created there)
    result_file : Path
        The file where the results were previously exported

    Raises
    ------
    FileNotFoundError
        If the results file does not exist
    """
    plot_dir = results_path / "plots"
    if not plot_dir.exists():
        plot_dir.mkdir()

    if not result_file.exists():
        raise FileNotFoundError("Result file not found")

    results = pd.read_csv(result_file, sep=";")

    if results.size == 0:
        logger.info("Eval files empty, no plots are being made")
        return

    metrics = ["Dice", "Connectivity", "Fragmentation", "Mean Symmetric Surface Distance"]

    for met in metrics:

        groups = results.groupby("fold")  # pylint: disable=no-member
        labels = list(groups.groups.keys())
        data = groups[met].apply(list).values

        plt.figure(figsize=(2 * len(data) + 5, 10))
        ax = plt.subplot(111)
        for i in ax.spines.values():
            i.set_linewidth(1)

        ax.set_title(f"{results_path.name} {met}", pad=20)
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(20)

        plt.boxplot(
            data,
            notch=False,
            showmeans=True,
            showfliers=True,
            vert=True,
            widths=0.9,
            patch_artist=True,
            labels=labels,
        )

        plt.savefig(plot_dir / (met.replace(" ", "") + ".png"), transparent=False)
        plt.close()
