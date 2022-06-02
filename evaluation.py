"""
Collection of functions to evaluate and plot the results.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import SimpleITK as sitk
from sklearn import metrics

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
    predictions: np.ndarray, std: np.ndarray, ground_truth: Any, map_dict: Dict[Any, int]
) -> Dict[str, float]:
    """Evaluate a classification task

    Parameters
    ----------
    predictions : np.ndarray
        The predictions to evaluate as np.array
    ground_truth : Any
        The ground truth
    map_dict : Dict[Any, int]
        The mapping of the previous categories to the classes

    Returns
    -------
    Dict[str, float]
        The resulting metric with one entry per metric
    """
    metrics_dict = {}
    # reverse the mapping dictionary to get the original labels
    map_dict_rev = {v: k for k, v in map_dict.items()}
    pred_mean = predictions.mean(axis=tuple(range(predictions.ndim - 1)))
    class_prediction = map_dict_rev[np.argmax(pred_mean)]
    metrics_dict["accuracy"] = int(class_prediction == ground_truth)
    metrics_dict["top_prediction"] = class_prediction
    metrics_dict["ground_truth"] = ground_truth
    for num, prob_val in enumerate(pred_mean):
        label = map_dict_rev[num]
        metrics_dict[f"probability_{label}"] = prob_val
    metrics_dict["std"] = std[np.argmax(pred_mean)]
    return metrics_dict


def calculate_classification_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    probabilities: np.ndarray,
    labels: List,
) -> Dict:
    """Calculate a few classification metrics. Those include:
    - accuracy
    - confusion_matrix
    - precision
    - recall
    - precision_mean (using micro average)
    - recall_mean (using micro average)
    - auc_ovo (using ovo for multi class)
    - auc_ovr (using ovr for multi class)

    Parameters
    ----------
    prediction : np.ndarray
        The predicted scores as 1d array
    ground_truth : np.ndarray
        The ground truth as 1d array
    probabilities : np.ndarray
        The probabilities as a 2d array
    labels : List
        The labels as list

    Returns
    -------
    Dict[str, Any]
        A dictionary with one entry for each metric
    """
    # treat it as strings
    ground_truth = ground_truth.astype(str)
    prediction = prediction.astype(str)
    assert np.allclose(prediction.shape, ground_truth.shape)

    metrics_dict = {}

    metrics_dict["accuracy"] = np.mean(prediction == ground_truth)

    confusion_matrix = metrics.confusion_matrix(ground_truth, prediction)
    metrics_dict["confusion_matrix"] = confusion_matrix
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    # replace nans (when it is undefined) with 0s
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    metrics_dict["precision"] = precision
    metrics_dict["recall"] = recall
    metrics_dict["precision_mean"] = metrics.precision_score(
        ground_truth, prediction, average="micro"
    )
    metrics_dict["recall_mean"] = metrics.recall_score(
        ground_truth, prediction, average="micro"
    )

    assert probabilities.shape[-1] == len(labels)
    # because of the average, the probabilities might not exactly add up to 1.
    prob_sum = probabilities.sum(axis=1)
    assert np.allclose(prob_sum, 1, atol=0.2)
    probabilities = (probabilities.T / prob_sum).T

    metrics_dict["auc_ovo"] = metrics.roc_auc_score(
        y_true=ground_truth, y_score=probabilities, labels=labels, multi_class="ovo"
    )
    metrics_dict["auc_ovr"] = metrics.roc_auc_score(
        y_true=ground_truth, y_score=probabilities, labels=labels, multi_class="ovr"
    )

    for k in [2, 3, 5]:
        if len(labels) > k:
            metrics_dict[f"top_{k}_accuracy"] = metrics.top_k_accuracy_score(
                k=k,
                y_true=ground_truth,
                y_score=probabilities,
                labels=labels,
            )

    return metrics_dict


def evaluate_regression(
    predictions: np.ndarray, ground_truth: float, map_dict: Dict[float, float]
) -> Dict[str, float]:
    """Evaluate a regression task, calculates the following metrics:
    - rmse (root mean square error)
    - rmse_rel (divided by the mean prediction)
    - mean_absolute_error
    - mean_prediction
    - mean_ground_truth
    - std (standard deviation of the prediction)

    Parameters
    ----------
    predictions : np.ndarray
        The predictions to evaluate as np.array
    ground_truth : float
        The ground truth
    map_dict: Dict[float float]
        The mapping between the network output (keys) and the input (values)

    Returns
    -------
    Dict[str, float]
        A dictionary with one entry for each metric
    """
    metrics_dict = {}
    mapping = scipy.interpolate.interp1d(list(map_dict.keys()), list(map_dict.values()))
    # make sure it is between 0 and 1
    pred = mapping(
        np.clip(predictions, np.min(list(map_dict.keys())), np.max(list(map_dict.keys())))
    )
    pred_mean = pred.mean()
    error = pred - ground_truth
    error_abs = np.abs(error)
    rmse = np.sqrt(np.mean(np.square(error)))
    metrics_dict["rmse"] = rmse
    metrics_dict["rmse_rel"] = rmse / pred_mean
    metrics_dict["mean_absolute_error"] = np.mean(error_abs)
    metrics_dict["mean_prediction"] = pred_mean
    metrics_dict["mean_ground_truth"] = np.mean(ground_truth)
    metrics_dict["std"] = pred.std()
    return metrics_dict


def calculate_regression_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> Dict:
    """_summary_

    Parameters
    ----------
    prediction : np.ndarray
        The prediction as 1d array
    ground_truth : np.ndarray
        The ground truth as 1d array

    Returns
    -------
    Dict
        _description_
    """
    ground_truth = ground_truth.astype(float)
    prediction = prediction.astype(float)
    assert np.allclose(prediction.shape, ground_truth.shape)

    pred_mean = prediction.mean()
    error = prediction - ground_truth
    error_abs = np.abs(error)
    rmse = np.sqrt(np.mean(np.square(error)))
    metrics_dict = {}

    metrics_dict["rmse"] = rmse
    metrics_dict["rmse_rel"] = rmse / pred_mean
    metrics_dict["mean_absolute_error"] = np.mean(error_abs)
    metrics_dict["max_absolute_error"] = np.max(error_abs)
    metrics_dict["min_absolute_error"] = np.min(error_abs)
    metrics_dict["std"] = prediction.std()

    return metrics_dict


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

    metric_names = [
        "Dice",
        "Connectivity",
        "Fragmentation",
        "Mean Symmetric Surface Distance",
    ]

    for met in metric_names:

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
