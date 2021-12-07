"""This module just contains the SegBasisNet.
"""
import logging
import os
from pathlib import Path
from typing import Callable, List, Union

import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from . import tf_utils
from .NetworkBasis import loss
from .NetworkBasis.metric import Dice
from .NetworkBasis.network import Network

from . import config as cfg

# configure logger
logger = logging.getLogger(__name__)

# it inherits the public methods
# pylint: disable=too-few-public-methods


class SegBasisNet(Network):
    """Basic network to perform segmentation."""

    def __init__(
        self,
        loss_name: str,
        is_training=True,
        do_finetune=False,
        model_path="",
        drop_out=(False, 0.2),
        regularize=(True, "L2", 0.00001),
        activation="relu",
        debug=False,
        **kwargs,
    ):

        # set tensorflow mixed precision policy (does not work together with gradient clipping)
        # policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        # tf.keras.mixed_precision.experimental.set_policy(policy)

        self.custom_objects = {"Dice": Dice}

        super().__init__(
            loss=loss_name,
            is_training=is_training,
            do_finetune=do_finetune,
            model_path=model_path,
            regularize=regularize,
            drop_out=drop_out,
            activation=activation,
            debug=debug,
            **kwargs,
        )

        # window size when applying the network
        self.window_size = None
        # number each input dimension (besides rank) should be divisible by (to avoid problems in maxpool layer)
        # this number should be determined by the network
        self.divisible_by = None

    def _set_up_inputs(self):
        """setup the inputs. Inputs are taken from the config file."""
        self.inputs["x"] = tf.keras.Input(
            shape=cfg.train_input_shape, batch_size=cfg.batch_size_train, dtype=cfg.dtype
        )
        self.options["out_channels"] = cfg.num_classes_seg

    def _get_loss(self, loss_name) -> Callable:
        """
        Returns loss depending on loss.

        loss should be in {'DICE', 'TVE', 'GDL', 'CEL', 'WCEL'}.

        Returns
        -------
        Callable
            The loss as tensorflow function
        """
        # many returns do not affect the readability
        # pylint: disable=too-many-return-statements

        if loss_name == "DICE":
            return loss.dice_loss

        if loss_name == "DICE-FNR":
            return loss.dice_with_fnr_loss

        if loss_name == "TVE":
            return loss.tversky_loss

        if loss_name == "GDL":
            return loss.generalized_dice_loss

        if loss_name == "GDL-FPNR":
            return loss.generalized_dice_with_fpr_fnr_loss

        if loss_name == "NDL":
            return loss.normalized_dice_loss

        if loss_name == "EDL":
            return loss.equalized_dice_loss

        if loss_name == "CEL":
            if self.options["out_channels"] > 2:
                return loss.categorical_cross_entropy_loss
            else:
                return loss.binary_cross_entropy_loss

        if loss_name == "ECEL":
            return loss.equalized_categorical_cross_entropy

        if loss_name == "NCEL":
            return loss.normalized_categorical_cross_entropy

        if loss_name == "WCEL":
            return loss.weighted_categorical_cross_entropy

        if loss_name == "ECEL-FNR":
            return loss.equalized_categorical_cross_entropy_with_fnr

        if loss_name == "WCEL-FPR":
            return loss.weighted_categorical_crossentropy_with_fpr_loss

        if loss_name == "GCEL":
            return loss.generalized_categorical_cross_entropy

        if loss_name == "CEL+DICE":
            return loss.categorical_cross_entropy_and_dice_loss

        raise ValueError(loss_name, "is not a supported loss function.")

    def _select_final_activation(self):
        # http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/
        if self.options["out_channels"] > 2 or self.options["loss"] in [
            "DICE",
            "TVE",
            "GDL",
        ]:
            # Dice, GDL and Tversky require SoftMax
            return "softmax"
        elif self.options["out_channels"] == 2 and self.options["loss"] in [
            "CEL",
            "WCEL",
            "GCEL",
        ]:
            return "sigmoid"
        else:
            raise ValueError(
                self.options["loss"],
                "is not a supported loss function or cannot combined with ",
                self.options["out_channels"],
                "output channels.",
            )

    def _build_model(self) -> tf.keras.Model:
        raise NotImplementedError("not implemented")

    # pylint: disable=arguments-differ
    def _run_train(
        self,
        logs_path: os.PathLike,
        folder_name: str,
        training_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        epochs: int = 10,
        l_r=0.001,
        optimizer="Adam",
        metrics=("dice", "acc", "meanIoU"),
        monitor="val_loss",
        monitor_mode="min",
        best_model_decay=0.7,
        early_stopping=False,
        patience_es=10,
        reduce_lr_on_plateau=False,
        patience_lr_plat=5,
        factor_lr_plat=0.5,
        visualization_dataset=None,
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
        logs_path : str
            The path for the output of the different networks
        folder_name : str
            This is used as the folder name, so the output is logs_path / folder_name
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
        metrics : Collection[Union[str, Callable]], optional
            The metrics that should be used a strings or Callables, by default ("dice", "acc", "meanIoU")
        monitor : str, optional
            The metric to monitor, used for early stopping and keeping the best model and lr reduction.
            Prefix val_ means that the metric from the validation dataset will be used, by default "val_loss"
        monitor_mode : str, optional
            The mode to use for monitoring the metric, min or max, by default min
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
            The factor by which the learing rate is multiplied at a plateau, by default 0.5
        visualization_dataset: SegBasisLoader, optional
            If provided, this dataset will be used to visualize the training results for debugging.
            Writing the images can take a bit, so only use this for debugging purposes.
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
        output_path = Path(logs_path) / folder_name

        # do summary
        self.model.summary(print_fn=logger.info)

        # set metrics
        metric_objects: List[Union[str, Callable]] = []
        for met in metrics:
            if met == "dice":
                metric_objects.append(Dice(name="dice", num_classes=cfg.num_classes_seg))
            elif met == "meanIoU":
                metric_objects.append(
                    tf.keras.metrics.MeanIoU(num_classes=cfg.num_classes_seg)
                )
            elif met == "fp":
                metric_objects.append(tf.keras.metrics.FalsePositives())
            elif met == "fn":
                metric_objects.append(tf.keras.metrics.FalseNegatives())
            elif met == "tn":
                metric_objects.append(tf.keras.metrics.TrueNegatives())
            elif met == "tp":
                metric_objects.append(tf.keras.metrics.TruePositives())
            elif met == "precision":
                metric_objects.append(tf.keras.metrics.Precision())
            elif met == "recall":
                metric_objects.append(tf.keras.metrics.Recall())
            elif met == "auc":
                metric_objects.append(tf.keras.metrics.AUC())
            # if nothing else is specified, just add it
            elif isinstance(met, str):
                metric_objects.append(met)

        # compile model
        self.model.compile(
            optimizer=self._get_optimizer(
                optimizer, l_r, 0, clipvalue=self.options["clipping_value"]
            ),
            loss=self.outputs["loss"],
            metrics=metric_objects,
            run_eagerly=debug,
        )

        if self.options["debug"]:
            self.model.run_eagerly = True

        # check the iterator sizes
        assert cfg.num_files is not None, "Number of files should be set"
        iter_per_epoch = cfg.samples_per_volume * cfg.num_files // cfg.batch_size_train
        assert iter_per_epoch > 0, "Steps per epoch is zero, lower the batch size"
        iter_per_vald = cfg.samples_per_volume * cfg.number_of_vald // cfg.batch_size_valid
        assert (
            iter_per_vald > 0
        ), "Steps per epoch is zero for the validation, lower the batch size"

        # define callbacks
        callbacks = []

        # to save the model
        model_dir = output_path / "models"
        if not model_dir.exists():
            model_dir.mkdir()

        # to save the best model
        cp_best_callback = tf_utils.KeepBestModel(
            filepath=model_dir / f"weights_best_{{epoch:03d}}-best{{{monitor}:1.3f}}.hdf5",
            save_weights_only=True,
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
                monitor=monitor, patience=patience_es, mode=monitor_mode, min_delta=0.005
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
                cooldown=5,
            )
            callbacks.append(lr_reduce_callback)

        # for tensorboard
        # only write gradients if there is a visualization dataset
        write_grads = visualization_dataset is not None
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
        )
        callbacks.append(tb_callback)

        # callback for hyperparameters
        hparams = self.get_hyperparameter_dict()
        # set additional paramters
        hparams["folder_name"] = folder_name
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

    def get_hyperparameter_dict(self):
        """This function reads the hyperparameters from options and writes them to a dict of
        hyperparameters, which can then be read using tensorboard.

        Returns
        -------
        dict
            the hyperparameters as a dictionary
        """
        # TODO: move into the individual networks
        hyperparameters = {
            "dimension": self.options["rank"],
            "drop_out_rate": self.options["drop_out"][1],
            "regularize": self.options["regularize"][0],
            "regularizer": self.options["regularize"][1],
            "regularizer_param": self.options["regularize"][2],
        }
        if "kernel_dim" in self.options:
            if isinstance(self.options["kernel_dims"], list):
                hyperparameters["kernel_dim"] = self.options["kernel_dims"][0]
            else:
                hyperparameters["kernel_dim"] = self.options["kernel_dims"]
        if "dilation_rate" in self.options:
            hyperparameters["dilation_rate_first"] = self.options["dilation_rate"][0]
        if self.options["regularize"]:
            if isinstance(self.options["regularizer"], tf.keras.regularizers.L2):
                hyperparameters["L2"] = self.options["regularizer"].get_config()["l2"]
        # add filters
        if "n_filters" in self.options:
            for i, f in enumerate(self.options["n_filters"]):
                hyperparameters[f"n_filters_{i}"] = f
        for opt in [
            "use_bias",
            "loss",
            "name",
            "batch_normalization",
            "activation",
            "use_cross_hair",
            "res_connect",
            "regularize",
            "use_bias",
            "skip_connect",
            "n_blocks",
            "backbone",
        ]:
            if opt in self.options:
                hyperparameters[opt] = self.options[opt]
        # check the types
        for key, value in hyperparameters.items():
            if type(value) in [list, tuple]:
                hyperparameters[key] = str(value)
        return hyperparameters

    def _run_apply(self, version, model_path, application_dataset, filename, apply_path):
        """Apply the network to test data. If the network is 2D, it is applied
        slice by slice. If it is 3D, it is applied to the whole images. If that
        runs out of memory, it is applied in patches in z-direction with the same
        size as used in training.

        Parameters
        ----------
        version : int or str
            The epoch, can be int or identifier (final for example)
        model_path : str
            Not used
        application_dataset : ApplyBasisLoader
            The dataset
        filename : str
            The file that is being processed, used to generate the new file name
        apply_path : str
            Where the files are written
        """

        if not os.path.exists(apply_path):
            os.makedirs(apply_path)

        # set the divisible by parameter
        application_dataset.divisible_by = self.divisible_by

        logger.debug("Load the image.")

        image_data = application_dataset(filename)

        # clear session
        tf.keras.backend.clear_session()

        logger.debug("Starting to apply the image.")

        # if 2D, run each layer as a batch, this should probably not run out of memory
        if self.options["rank"] == 2:
            predictions = []
            window_shape = [1] + cfg.train_input_shape[:2]
            overlap = [0, 15, 15]
            image_data_patches = application_dataset.get_windowed_test_sample(
                image_data, window_shape, overlap
            )
            # remove z-dimension
            image_data_patches = image_data_patches.reshape([-1] + cfg.train_input_shape)
            # turn into batches with last batch being not a full one
            batch_rest = image_data_patches.shape[0] % cfg.batch_size_train
            batch_shape = (-1, cfg.batch_size_train) + image_data_patches.shape[-3:]
            if batch_rest != 0:
                image_data_batched = image_data_patches[:-batch_rest].reshape(batch_shape)
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
            logger.debug("Applied in 2D using the original size of each slice.")
        # otherwise, just run it
        else:
            try:
                # if the window size exists, trigger the exception
                if self.window_size is not None:
                    raise tf.errors.ResourceExhaustedError(None, None, "trigger exception")
                probability_map = self.model(image_data, training=False)
            except tf.errors.ResourceExhaustedError:
                # try to reduce the patch size to a size that works
                if self.window_size is None:
                    # initial size is the image size
                    self.window_size = np.array(image_data.shape[1:4])
                    # try to find the best patch size (do not use more than 10 steps)
                    for i in range(10):
                        # reduce the window size
                        # reduce z if it is larger than the training shape
                        if self.window_size[0] > cfg.train_input_shape[0]:
                            red = 0
                        # otherwise, lower the larger of the two other dimensions
                        elif self.window_size[1] >= self.window_size[2]:
                            red = 1
                        else:
                            red = 2
                        # reduce the size
                        self.window_size[red] = self.window_size[red] // 2
                        # make it divisible by 16
                        self.window_size[red] = (
                            int(np.ceil(self.window_size[red] / 16)) * 16
                        )
                        try:
                            test_data_shape = (
                                (1,) + tuple(self.window_size) + (image_data.shape[-1],)
                            )
                            test_data = np.random.rand(*test_data_shape)
                            self.model(test_data, training=False)
                            self.model.predict(x=test_data, batch_size=1)
                        except tf.errors.ResourceExhaustedError:
                            logger.debug(
                                "Applying failed for window size %s in step %i.",
                                self.window_size,
                                i,
                            )
                        else:
                            # if it works, break the cycle and reduce once more to be sure
                            # reduce the window size
                            # reduce z if it is larger than the training shape
                            if self.window_size[0] > cfg.train_input_shape[0]:
                                red = 0
                            # otherwise, lower the larger of the two other dimensions
                            elif self.window_size[1] >= self.window_size[2]:
                                red = 1
                            else:
                                red = 2
                            # reduce the size
                            self.window_size[red] = self.window_size[red] // 2
                            # make it divisible by 16
                            self.window_size[red] = (
                                int(np.ceil(self.window_size[red] / 16)) * 16
                            )
                            break
                # get windowed samples
                image_data_patches = application_dataset.get_windowed_test_sample(
                    image_data, self.window_size, overlap=[5, 15, 15]
                )
                probability_patches = []
                # apply
                for x in image_data_patches:
                    result = self.model.predict(x=x, batch_size=1)
                    probability_patches.append(result)
                probability_map = application_dataset.stitch_patches(probability_patches)
            else:
                logger.debug("Applied using the original size of the image.")

        # remove padding
        probability_map = application_dataset.remove_padding(probability_map)

        # remove the batch dimension
        if self.options["rank"] == 3:
            probability_map = probability_map[0]

        # get the labels
        predicted_labels = np.argmax(probability_map, -1)

        # get the processed image
        orig_processed = application_dataset.get_processed_image(filename)

        predicted_label_img = sitk.GetImageFromArray(predicted_labels)
        # copy info
        predicted_label_img.CopyInformation(orig_processed)

        logger.debug("Predicted labels were calculated.")

        # get the original image
        original_image = application_dataset.get_original_image(filename)

        # resample to the original file
        predicted_label_orig = sitk.Resample(
            image1=predicted_label_img,
            referenceImage=original_image,
            interpolator=sitk.sitkNearestNeighbor,
            outputPixelType=sitk.sitkUInt8,
            useNearestNeighborExtrapolator=False,
        )

        # write resampled file
        name = Path(filename).name
        pred_path = Path(apply_path) / f"prediction-{name}-{version}{cfg.file_suffix}"
        sitk.WriteImage(predicted_label_orig, str(pred_path.resolve()))

        if cfg.write_probabilities:
            # turn probabilities into an image
            probability_map_img = sitk.GetImageFromArray(probability_map)
            probability_map_img.CopyInformation(orig_processed)
            f_name = (
                Path(apply_path)
                / f"prediction-{name}-{version}_probabilities{cfg.file_suffix}"
            )
            sitk.WriteImage(probability_map_img, str(f_name.resolve()))

        if cfg.write_intermediaries:
            # write the preprocessed image
            proc_path = Path(apply_path) / f"sample-{name}-preprocessed{cfg.file_suffix}"
            sitk.WriteImage(orig_processed, str(proc_path.resolve()))

            # write the labels for the preprocessed image
            pred_res_path = (
                Path(apply_path)
                / f"prediction-{name}-{version}-preprocessed{cfg.file_suffix}"
            )
            sitk.WriteImage(
                sitk.Cast(predicted_label_img, sitk.sitkUInt8), str(pred_res_path.resolve())
            )

        logger.debug("Images were exported.")
