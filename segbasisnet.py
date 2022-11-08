"""This module just contains the SegBasisNet.
"""
import collections
import logging
import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.optimizers import schedules

from . import config as cfg
from . import loss, tf_utils, utils
from .metric import NMI, Dice

# configure logger
logger = logging.getLogger(__name__)


class SegBasisNet:
    """Basic network to perform segmentation.

    Inheriting classes should set:
    - name : the name of the model
    - _build_model : This function will actually build the model
    - get_hyperparameter_dict : all relevant hyperparameters for the model
    - self.divisible_by should be set somewhere (some models need to have the input be
      divisible by a certain number, for example because of maxpool layers)

    Parameters
    ----------
    loss_name : str
        The name of the loss to use. If multiple tasks are performed, it can also
        be a dict with the task name as keys and a list of losses as values
    tasks : OrderedDict[str, str], optional
        The tasks that should be performed, loss and metrics will be selected accordingly
    """

    name: str

    def __init__(
        self,
        loss_name: Union[dict, str, Callable],
        tasks: Optional[OrderedDict[str, str]] = None,
        is_training=True,
        do_finetune=False,
        model_path="",
        regularize=(True, "L2", 0.00001),
        **kwargs,
    ):

        # set tensorflow mixed precision policy (does not work together with gradient clipping)
        # policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        # tf.keras.mixed_precision.experimental.set_policy(policy)

        self.custom_objects = {"Dice": Dice, "NMI": NMI}
        if tasks is None:
            tasks = collections.OrderedDict({"seg": "segmentation"})
        if not isinstance(tasks, collections.OrderedDict):
            raise ValueError("tasks should be an ordered dict")
        self.tasks = tuple(tasks.values())
        self.task_names = tuple(tasks.keys())

        self.inputs: Dict[str, tf.keras.Input] = {}
        self.outputs = {}
        # output is a dictionary containing logits, probabilities, loss and predictions
        self.variables: Dict[str, Any] = {}
        self.options: Dict[str, Any] = {}
        self.options["is_training"] = is_training
        self.options["do_finetune"] = do_finetune
        self.options["regularize"] = regularize

        if not self.options["is_training"] or (
            self.options["is_training"] and self.options["do_finetune"]
        ):
            if model_path == "":
                raise ValueError("Model Path cannot be empty for Finetuning or Inference!")
        else:
            if model_path != "":
                logger.warning("Caution: argument model_path is ignored in training!")

        self.options["model_path"] = model_path
        if not hasattr(self, "tasks"):
            self.tasks = ("segmentation",)

        # write other kwargs to options
        for key, value in kwargs.items():
            self.options[key] = value

        # window size when applying the network
        self.window_size = None
        # number each input dimension (besides rank) should be divisible by (to avoid problems in maxpool layer)
        # this number should be determined by the network
        self.divisible_by = 1
        # extra compile options, which can be changed in subclasses
        self._compile_options: Dict[str, Any] = {}

        if self.options["is_training"] and not self.options["do_finetune"]:
            self.set_up_inputs()
            self.options["regularizer"] = self._get_reg()
            # if training build everything according to parameters
            if hasattr(self.inputs["x"], "shape"):
                self.options["in_channels"] = self.inputs["x"].shape.as_list()[-1]
                # self.options['out_channels'] is set elsewhere, but required
                # input number of dimensions (without channels and batch size)
                self.options["rank"] = len(self.inputs["x"].shape) - 2
            else:
                raise AttributeError("Input has no shape")

            # tf.summary.trace_on(graph=True, profiler=False)
            self.outputs["loss"], self.options["loss_name"] = self._get_task_losses(
                loss_name
            )
            self.model = self._build_model()
        else:
            # for finetuning load net from file
            # tf.summary.trace_on(graph=True, profiler=False)
            self._load_net()
            self.outputs["loss"], self.options["loss_name"] = self._get_task_losses(
                loss_name
            )

    def set_up_inputs(self):
        """setup the inputs. Inputs are taken from the config file."""
        ndim = len(cfg.train_input_shape) - 1
        input_shape = [None] * ndim + cfg.train_input_shape[-1:]
        self.inputs["x"] = tf.keras.Input(
            shape=input_shape,
            batch_size=None,
            dtype=cfg.dtype,
            name="input",
        )
        self.options["out_channels"] = cfg.num_classes_seg

    def _build_model(self) -> tf.keras.Model:
        raise NotImplementedError("not implemented")

    def _load_net(self):
        # This loads the keras network and the first checkpoint file
        self.model: tf.keras.Model = tf.keras.models.load_model(
            self.options["model_path"], compile=False, custom_objects=self.custom_objects
        )

        epoch = Path(self.options["model_path"]).name.split("-")[-1]
        self.variables["epoch"] = epoch

        if self.options["is_training"] is False:
            self.model.trainable = False
        self.options["rank"] = len(self.model.inputs[0].shape) - 2
        self.options["in_channels"] = self.model.inputs[0].shape.as_list()[-1]
        self.options["out_channels"] = self.model.outputs[0].shape.as_list()[-1]
        logger.info("Model was loaded")

    def _get_reg(self) -> tf.keras.regularizers.Regularizer:
        if self.options["regularize"][0]:
            if self.options["regularize"][1] == "L1":
                regularizer = tf.keras.regularizers.l1(self.options["regularize"][2])
            elif self.options["regularize"][1] == "L2":
                regularizer = tf.keras.regularizers.l2(self.options["regularize"][2])
            else:
                raise ValueError(
                    self.options["regularize"][1], "is not a supported regularizer ."
                )
        else:
            regularizer = None

        return regularizer

    def _get_task_losses(self, loss_input: Union[str, dict, object, Iterable]):
        if isinstance(loss_input, str):
            loss_obj = self.get_loss(loss_input)
            loss_name = loss_input
        elif hasattr(loss_input, "__call__"):
            if hasattr(loss_input, "__name__"):
                loss_name = loss_input.__name__
            else:
                loss_name = type(loss_input).__name__
            loss_obj = loss_input
        elif isinstance(loss_input, dict):
            loss_name = "Multitask"
            loss_obj = tuple(self.get_loss(loss_input[t], t) for t in self.tasks)
        elif isinstance(loss_input, Iterable):
            loss_name = "-".join(str(l) for l in loss_input)
            loss_obj = tuple(self.get_loss(l) for l in loss_input)
        else:
            raise ValueError(
                "Incompatible loss, it should be a Callable, string, dict or Iterable."
            )
        return loss_obj, loss_name

    def get_loss(self, loss_name: str, task="segmentation") -> Callable:
        """
        Returns loss depending on loss.

        just look at the function to see the allowed losses

        Parameters
        ----------
        loss_name : str
            The name of the loss
        task : str, optional
            The task being performed, by default segmentation.

        Returns
        -------
        Callable
            The loss as tensorflow function
        """
        if isinstance(loss_name, (tuple, list)):
            losses = tuple(self.get_loss(l) for l in loss_name)

            def custom_loss(y_true, y_pred):
                return sum(l(y_true, y_pred) for l in losses)

            return custom_loss
        if "loss_parameters" in self.options:
            loss_parameters = self.options["loss_parameters"].get(loss_name, None)
        else:
            loss_parameters = None
        return loss.get_loss(loss_name, loss_parameters)

    def get_lr_scheduler(
        self, schedule_type: str, initial_rate: float, final_rate: float
    ) -> schedules.LearningRateSchedule:
        """Get the learning rate scheduler, the decay rate is calculated using
        the initial and final learning rate. So far, only exponential and
        exponential_half is implemented

        Parameters
        ----------
        schedule_type : str
            The type of scheduler
        initial_rate : float
            The initial learning rate
        final_rate : float
            The final learning rate (at the end of training)

        Returns
        -------
        schedules.LearningRateSchedule
            The scheduler which can be passed to the optimizer
        """
        assert cfg.num_files is not None
        iter_per_epoch = cfg.samples_per_volume * cfg.num_files // cfg.batch_size_train
        n_epochs = self.options["n_epochs"]
        if schedule_type == "exponential":
            decay_rate = (final_rate / initial_rate) ** (1 / n_epochs)
            lr_schedule = schedules.ExponentialDecay(
                initial_learning_rate=initial_rate,
                decay_steps=iter_per_epoch,
                decay_rate=decay_rate,
            )
        elif schedule_type == "exponential_half":
            decay_rate = (final_rate / initial_rate) ** (1 / (n_epochs / 2))
            lr_schedule = tf_utils.ExponentialDecayMin(
                initial_learning_rate=initial_rate,
                decay_steps=iter_per_epoch,
                decay_rate=decay_rate,
                final_rate=final_rate,
            )
        else:
            raise ValueError(f"LR scheduler {schedule_type} unknown")
        assert np.isclose(lr_schedule(n_epochs * iter_per_epoch), final_rate)
        return lr_schedule

    def plot_model(self, save_dir: Path):
        """Plot the model to the save dir

        Parameters
        ----------
        save_dir : Path
            Where to save the model
        """
        tf.keras.utils.plot_model(
            self.model,
            to_file=save_dir / "model.png",
        )
        tf.keras.utils.plot_model(
            self.model, to_file=save_dir / "model_with_shapes.png", show_shapes=True
        )

    # pylint: disable=arguments-differ
    def train(
        self,
        base_output_path: os.PathLike,
        folder_name: str,
        training_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        epochs: int = 10,
        l_r=0.001,
        optimizer="Adam",
        metrics: Optional[Dict[str, Collection[Union[str, Callable]]]] = None,
        monitor="val_loss",
        monitor_mode="min",
        save_best_only=True,
        best_model_decay=0.7,
        early_stopping=False,
        patience_es=10,
        reduce_lr_on_plateau=False,
        patience_lr_plat=5,
        factor_lr_plat=0.5,
        write_tensorboard=True,
        visualization_dataset=None,
        write_grads=False,
        visualize_labels=True,
        write_graph=True,
        debug=False,
        finetune_epoch=None,
        finetune_layers=None,
        finetune_lr=None,
        **kwargs,
    ):
        """Run the training using the keras.Model.fit interface with a lot of callbacks.

        Parameters
        ----------
        base_output_path : str
            The path for the output of the different networks
        folder_name : str
            This is used as the folder name, so the output is base_output_path / folder_name
        training_dataset : Tensorflow dataset
            The dataset for training, you can use the SegBasisLoader ofr this (call it)
        validation_dataset : Tensorflow dataset
            The dataset for validation, you can use the SegBasisLoader ofr this (call it)
        epochs : int
            The number of epochs
        l_r : float, optional
            The learning rate, by default 0.001
        optimizer : str, optional
            The name of the optimizer, by default 'Adam'
        metrics : Dict[str, Collection[Union[str, Callable]]], optional
            The metrics are a dict with the task as key and then the metrics as values.
            The metrics that should be used a strings or Callables, by default ("dice", "acc", "meanIoU")
        monitor : str, optional
            The metric to monitor, used for early stopping and keeping the best model and lr reduction.
            Prefix val_ means that the metric from the validation dataset will be used, by default "val_loss"
        monitor_mode : str, optional
            The mode to use for monitoring the metric, min or max, by default min
        save_best_only : bool, optional
        If only the best model(s) should be saved, by default True
        best_model_decay : float, optional
            The decay rate used for averaging the metric when saving the best model,
            by default 0.7, None means no moving average
        early_stopping : bool, optional
            If early stopping should be used, by default False
        patience_es : int, optional
            The default patience for early stopping, by default 10
        reduce_lr_on_plateau : bool, optional
            If the learning rate should be reduced on a plateau, by default False
        patience_lr_plat : int, optional
            The patience before reducing the learning rate, by default 5
        factor_lr_plat : float, optional
            The factor by which the learning rate is multiplied at a plateau, by default 0.5
        write_tensorboard: bool, optional
            If the tensorboard callback should be used, by default True
        visualization_dataset: SegBasisLoader, optional
            If provided, this dataset will be used to visualize the training results and input images.
            Writing the images can take a bit, so it is only done every 5 epochs.
        write_grads: bool, optional
            If true, gradient histograms will be written, by default False. Can take a while,
            so best for debugging
        visualize_labels: bool, optional
            If the labels should be visualized as images, by default True
        write_graph : bool, optional
            Controls if a graph should be written, can be used than only the first fold will
            get a graph, to prevent cluttering the output.
        debug : bool, optional
            build the network in debug mode and run it eagerly, by default false
        finetune_epoch : int, optional
            At which epoch fine-tuning should be enabled, if None, no finetuning will be done, by default None
        finetune_layers : str or list, optional
            Which layers should be finetuned. This can either be a list of names or all,
            which enables training on all layers besides batchnorm layers, by default None
        finetune_lr : float, optional
            If not None, this rate will be set after enabling the finetuning, by default None
        """

        # set path
        output_path = Path(base_output_path) / folder_name

        # to save the model
        model_dir = output_path / "models"
        if not model_dir.exists():
            model_dir.mkdir()

        # do summary
        self.model.summary(print_fn=logger.info)

        model_image_dir = model_dir / "images"
        if not model_image_dir.exists():
            model_image_dir.mkdir()
        self.plot_model(model_image_dir)

        if metrics is None:
            metrics = {
                "segmentation": ("dice", "acc", "meanIoU"),
                "classification": ("precision", "recall", "auc"),
                "discriminator-classification": (),
                "regression": ("rmse",),
                "discriminator-regression": (),
                "autoencoder": ("rmse", "nmi"),
            }

        metric_objects = self.get_task_metrics(metrics, self.tasks)

        if isinstance(l_r, (list, tuple)):
            l_r = self.get_lr_scheduler(*l_r)

        # compile model
        self.model.compile(
            optimizer=tf_utils.get_optimizer(
                optimizer, l_r, clipvalue=self.options["clip_value"]
            ),
            loss=self.outputs["loss"],
            metrics=metric_objects,
            run_eagerly=debug,
            **self._compile_options,
        )

        # check the iterator sizes
        assert cfg.num_files is not None, "Number of files should be set"
        iter_per_epoch = cfg.samples_per_volume * cfg.num_files // cfg.batch_size_train
        assert iter_per_epoch > 0, "Steps per epoch is zero, lower the batch size"
        iter_per_vald = int(
            np.ceil(cfg.samples_per_volume * cfg.number_of_vald / cfg.batch_size_valid)
        )
        assert (
            iter_per_vald > 0
        ), "Steps per epoch is zero for the validation, lower the batch size"

        # define callbacks
        callbacks = []

        # to save the best model
        if save_best_only:
            model_save_name = f"weights_best_{{epoch:03d}}-best{{{monitor}:1.5f}}.hdf5"
        else:
            model_save_name = "weights_{epoch:03d}.hdf5"
        cp_best_callback = tf_utils.KeepBestModel(
            filepath=model_dir / model_save_name,
            save_weights_only=True,
            save_best_only=save_best_only,
            verbose=0,
            save_freq="epoch",
            monitor=monitor,
            mode=monitor_mode,
            decay=best_model_decay,
        )
        callbacks.append(cp_best_callback)

        # early stopping
        if early_stopping:
            es_callback = tf.keras.callbacks.EarlyStopping(
                monitor=monitor, patience=patience_es, mode=monitor_mode, min_delta=5e-5
            )
            callbacks.append(es_callback)

        # reduce learning rate on plateau
        if reduce_lr_on_plateau:
            lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                patience=patience_lr_plat,
                mode=monitor_mode,
                factor=factor_lr_plat,
                verbose=1,
                min_lr=1e-6,
            )
            callbacks.append(lr_reduce_callback)

        if write_tensorboard:
            # ignore the latent dimension for the autoencoder
            if "autoencoder" in self.tasks and len(self.model.outputs) > 1:
                ignore = list(range(1, len(self.model.outputs)))
            else:
                ignore = None
            # for tensorboard
            tb_callback = tf_utils.CustomTBCallback(
                output_path / "logs",
                update_freq="epoch",
                profile_batch=(2, 12),
                histogram_freq=1,
                embeddings_freq=0,
                write_grads=write_grads,
                write_graph=write_graph,
                visualization_dataset=visualization_dataset,
                visualization_frequency=1,
                write_labels=visualize_labels,
                ignore=ignore,
            )
            callbacks.append(tb_callback)

        # callback for hyperparameters
        hparams = self.get_hyperparameter_dict()
        # set additional parameters
        hparams["folder_name"] = folder_name
        hparams["folder_parent_name"] = output_path.parent.name
        hp_callback = hp.KerasCallback(str(output_path / "logs" / "train"), hparams)
        callbacks.append(hp_callback)

        # callback to write csv data
        csv_callback = tf.keras.callbacks.CSVLogger(
            filename=output_path / "training.csv", separator=";"
        )
        callbacks.append(csv_callback)

        # callback for switch in trainable layers
        if finetune_epoch is not None:
            ft_callback = tf_utils.FinetuneLayers(
                to_activate=finetune_layers, epoch=finetune_epoch, learning_rate=finetune_lr
            )
            callbacks.append(ft_callback)

        if "CLUSTER" in os.environ:
            verbosity = 2
        else:
            verbosity = 1

        # do the training
        self.model.fit(
            x=training_dataset,
            epochs=epochs,
            verbose=verbosity,
            validation_data=validation_dataset,
            validation_freq=1,
            steps_per_epoch=iter_per_epoch,
            validation_steps=iter_per_vald,
            callbacks=callbacks,
        )
        print("Saving the final model.")
        self.model.save(model_dir / "model-final", save_format="tf")
        # save the best model
        best_val = None
        best_weights = None
        for val, weights in cp_best_callback.best_checkpoints.items():
            if best_val is None:
                best_val = val
                best_weights = weights
            elif val > best_val:
                best_val = val
                best_weights = weights
        self.model.load_weights(best_weights)
        print("Saving the best model.")
        self.model.save(model_dir / "model-best", save_format="tf")
        print("Saving finished.")

    def get_task_metrics(self, metrics: Dict, tasks: Collection[str]) -> List[Tuple[Any]]:
        """Get the metrics for the individual tasks as list of tuple

        Parameters
        ----------
        metrics : Dict
            The metrics as dict, the keys are the tasks nad the values are lists of metrics
        tasks : Collection[str]
            The tasks that should be used

        Returns
        -------
        list
            A list with an entry for each task with the metric objects
        """
        # set metrics
        metric_objects: List[Collection[Union[str, Callable]]] = []
        for t_name in tasks:
            metric_objects.append(
                tuple(self.get_metric(m, t_name) for m in metrics[t_name])
            )
        return metric_objects

    def get_metric(self, metric, task="segmentation") -> Union[Callable, str]:
        """Get the metric as callable object from the name

        Parameters
        ----------
        metric : str
            The name of the metric
        task : str, optional
            The task, depending on the metric, it might be necessary to adjust
            it depending on the task, by default "segmentation"

        Returns
        -------
        Callable | str
            The metric as callable or str, which is understood by tensorflow
        """
        nmi_params = {}
        if "loss_parameters" in self.options:
            nmi_params = self.options["loss_parameters"].get("NMI", {})
        metrics = {
            "dice": lambda: Dice(name="dice", num_classes=cfg.num_classes_seg),
            "nmi": lambda: NMI(name="nmi", **nmi_params),
            "meanIoU": lambda: tf.keras.metrics.MeanIoU(num_classes=cfg.num_classes_seg),
            "fp": tf.keras.metrics.FalsePositives,
            "fn": tf.keras.metrics.FalseNegatives,
            "tn": tf.keras.metrics.TrueNegatives,
            "tp": tf.keras.metrics.TruePositives,
            "precision": tf.keras.metrics.Precision,
            "recall": tf.keras.metrics.Recall,
            "acc": tf.keras.metrics.Accuracy,
            "auc": tf.keras.metrics.AUC,
            "rmse": tf.keras.metrics.RootMeanSquaredError,
        }
        if metric in metrics:
            return metrics[metric]()
        # if nothing else is specified, just add it
        elif isinstance(metric, str):
            return metric
        else:
            raise ValueError(f"Metric {metric} cannot be processed.")

    def get_hyperparameter_dict(self):
        """This function reads the hyperparameters from options and writes them to a dict of
        hyperparameters, which can then be read using tensorboard.

        Returns
        -------
        dict
            the hyperparameters as a dictionary
        """
        hyp = {
            "dimension": self.options.get("rank"),
            "regularize": self.options["regularize"][0],
            "regularizer": self.options["regularize"][1],
            "regularizer_param": self.options["regularize"][2],
            "loss": self.options["loss_name"],
        }
        hyperparameters = {key: str(value) for key, value in hyp.items()}
        return hyperparameters

    def apply(self, version, application_dataset, filename, apply_path):
        """Apply the network to test data. If the network is 2D, it is applied
        slice by slice. If it is 3D, it is applied to the whole images. If that
        runs out of memory, it is applied in patches in z-direction with the same
        size as used in training.

        Parameters
        ----------
        version : int or str
            The epoch, can be int or identifier (final for example)
        application_dataset : ApplyLoader
            The dataset
        filename : str
            The file that is being processed, used to generate the new file name
        apply_path : str
            Where the files are written
        """

        output = self.get_network_output(application_dataset, filename)
        apply_path = Path(apply_path)

        name = Path(filename).name
        res_name = f"prediction-{name}-{version}"

        for num, (out, task_name) in enumerate(zip(output, self.task_names)):
            if task_name in ("regression", "classification"):
                output[num] = out.squeeze()

        # export the images
        for out, tsk, task_name in zip(output, self.tasks, self.task_names):
            # remove padding
            if tsk in ("segmentation", "autoencoder"):
                if self.options["rank"] == 2:
                    out = application_dataset.remove_padding(out)
                else:
                    raise NotImplementedError()
            pred_img = utils.output_to_image(
                output=out,
                task=tsk,
                processed_image=application_dataset.get_processed_image(filename),
                original_image=application_dataset.get_original_image(filename),
            )
            new_image_path = apply_path / f"{res_name}_{task_name}{cfg.file_suffix}"
            sitk.WriteImage(pred_img, str(new_image_path.resolve()))

        # save the output as npz with task names as arguments
        output_path_npz = Path(apply_path) / f"{res_name}.npz"
        assert len(output) == len(self.task_names)
        # do not save the raw output, it is too big,
        utils.export_npz(output, self.tasks, self.task_names, output_path_npz)

    def get_network_output(self, application_dataset, filename: str) -> List[np.ndarray]:
        """Get the output of the network for a given example

        Parameters
        ----------
        application_dataset : Callable
            The dataset for the application, when called with the filename, it
            should produce slices in 2D and the whole image in 3D
        filename : str
            The filenames used as ID for the dataset

        Returns
        -------
        List[np.ndarray]
            List of results as numpy arrays. The length is the number of model outputs
        """
        n_outputs = len(self.model.outputs)

        # set the divisible by parameter
        application_dataset.divisible_by = self.divisible_by

        if self.options["rank"] == 2:
            image_data = application_dataset(filename)
            assert image_data.ndim == 4, "Image should have 4 dimensions"

            if self.model.output.shape[1:3].is_fully_defined():
                predictions = []
                window_shape = [1] + cfg.train_input_shape[:2]
                overlap = [0, 15, 15]
                image_data_patches = application_dataset.get_windowed_test_sample(
                    image_data, window_shape, overlap
                )
                # remove z-dimension
                image_data_patches = image_data_patches.reshape(
                    [-1] + cfg.train_input_shape
                )
                # turn into batches with last batch being not a full one
                batch_rest = image_data_patches.shape[0] % cfg.batch_size_train
                batch_shape = (-1, cfg.batch_size_train) + image_data_patches.shape[-3:]
                if batch_rest != 0:
                    image_data_batched = image_data_patches[:-batch_rest].reshape(
                        batch_shape
                    )
                    last_batch_shape = (batch_rest,) + image_data_patches.shape[-3:]
                    last_batch = image_data_patches[-batch_rest:].reshape(last_batch_shape)
                else:
                    image_data_batched = image_data_patches.reshape(batch_shape)
                for x in image_data_batched:
                    pred = self.model(x, training=False)
                    predictions.append(pred)
                if batch_rest != 0:
                    pred = self.model(last_batch, training=False)
                    predictions.append(pred)
                # concatenate
                probability_patches = np.concatenate(predictions)
                probability_map = application_dataset.stitch_patches(probability_patches)
                output = [probability_map]
            else:
                # reduce batch size according to data size
                batch_size = int(
                    np.prod(cfg.train_input_shape)
                    / np.prod(image_data.shape[1:])
                    * cfg.batch_size_train
                )
                image_data_batched = [
                    image_data[i : i + batch_size]
                    for i in range(0, image_data.shape[0], batch_size)
                ]
                results = []
                for sample in image_data_batched:
                    res = self.model(sample)
                    # make sure the result is a tuple
                    if not isinstance(res, tuple):
                        res = (res,)
                    # convert to numpy
                    res_np = tuple(r.numpy() for r in res)
                    results.append(res_np)

                # separate into multiple lists
                output_lists = [[row[out] for row in results] for out in range(n_outputs)]
                # and concatenate them
                output = [np.concatenate(out, axis=0) for out in output_lists]
        else:
            raise NotImplementedError("Only implemented for 2D")
        return output
