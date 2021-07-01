"""This module just contains the SegBasisNet.
"""
import logging
import os
from pathlib import Path
from typing import Callable

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

# TODO: remove dependence on Network or make it more abstract, it hardly does anything.
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

        self.inputs = {}
        self.outputs = {}
        self.variables = {}
        self.options = {}
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

        self.options["debug"] = debug

        # write other kwargs to options
        for key, value in kwargs.items():
            self.options[key] = value

        if self.options["is_training"] and not self.options["do_finetune"]:
            self._set_up_inputs()
            self.options[
                "drop_out"
            ] = drop_out  # if [0] is True, dropout is added to the graph with keep_prob [2]
            self.options["regularizer"] = self._get_reg()
            self.options["activation"] = activation
            # if training build everything according to parameters
            self.options["in_channels"] = self.inputs["x"].shape.as_list()[-1]
            # rank is the input number of dimensions (without channels and batch size)
            self.options["rank"] = len(self.inputs["x"].shape) - 2

            self.options["loss"] = loss_name
            self.outputs["loss"] = self._get_loss()
            self.model = self._build_model()
            self.model._name = self.get_name()
        else:
            # for finetuning load net from file
            self._load_net()
            self.options["loss"] = loss_name
            self.outputs["loss"] = self._get_loss()

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

    def _get_loss(self) -> Callable:
        """
        Returns loss depending on `self.options['loss']``.

        self.options['loss'] should be in {'DICE', 'TVE', 'GDL', 'CEL', 'WCEL'}.
        Returns
        -------
        Callable
            The loss as tensorflow function
        """
        # many returns do not affect the readability
        # pylint: disable=too-many-return-statements

        if self.options["loss"] == "DICE":
            return loss.dice_loss

        if self.options["loss"] == "DICE-FNR":
            return loss.dice_with_fnr_loss

        if self.options["loss"] == "TVE":
            return loss.tversky_loss

        if self.options["loss"] == "GDL":
            return loss.generalized_dice_loss

        if self.options["loss"] == "GDL-FPNR":
            return loss.generalized_dice_with_fpr_fnr_loss

        if self.options["loss"] == "NDL":
            return loss.normalized_dice_loss

        if self.options["loss"] == "EDL":
            return loss.equalized_dice_loss

        if self.options["loss"] == "WDL":
            return loss.weighted_dice_loss

        if self.options["loss"] == "CEL":
            if self.options["out_channels"] > 2:
                return loss.categorical_cross_entropy_loss
            else:
                return loss.binary_cross_entropy_loss

        if self.options["loss"] == "ECEL":
            return loss.equalized_categorical_cross_entropy

        if self.options["loss"] == "NCEL":
            return loss.normalized_categorical_cross_entropy

        if self.options["loss"] == "WCEL":
            return loss.weighted_categorical_cross_entropy

        if self.options["loss"] == "ECEL-FNR":
            return loss.equalized_categorical_cross_entropy_with_fnr

        if self.options["loss"] == "WCEL-FPR":
            return loss.weighted_categorical_crossentropy_with_fpr_loss

        if self.options["loss"] == "GCEL":
            return loss.generalized_categorical_cross_entropy

        if self.options["loss"] == "CEL+DICE":
            return loss.categorical_cross_entropy_and_dice_loss

        raise ValueError(self.options["loss"], "is not a supported loss function.")

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
        logs_path,
        folder_name,
        training_dataset,
        validation_dataset,
        epochs,
        l_r=0.001,
        optimizer="Adam",
        early_stopping=False,
        patience_es=10,
        reduce_lr_on_plateau=False,
        patience_lr_plat=5,
        factor_lr_plat=0.5,
        visualization_dataset=None,
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

        # compile model
        self.model.compile(
            optimizer=self._get_optimizer(
                optimizer, l_r, 0, clipvalue=self.options["clipping_value"]
            ),
            loss=self.outputs["loss"],
            metrics=[
                Dice(name="dice", num_classes=cfg.num_classes_seg),
                "acc",
                tf.keras.metrics.MeanIoU(num_classes=cfg.num_classes_seg),
            ],
            run_eagerly=debug,
        )

        if self.options["debug"]:
            self.model.run_eagerly = True

        # check the iterator sizes
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
            filepath=model_dir / "weights_best_{epoch:03d}-best{val_dice:1.3f}.hdf5",
            save_weights_only=True,
            verbose=0,
            save_freq="epoch",
            save_best_only=True,
            monitor="val_dice",
            mode="max",
        )
        callbacks.append(cp_best_callback)

        # backup and restore
        backup_dir = model_dir / "backup"
        if not backup_dir.exists():
            backup_dir.mkdir()
        backup_callback = tf.keras.callbacks.experimental.BackupAndRestore(
            backup_dir=backup_dir,
        )
        callbacks.append(backup_callback)

        # early stopping
        if early_stopping:
            es_callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_dice", patience=patience_es, mode="max", min_delta=0.005
            )
            callbacks.append(es_callback)

        # reduce learning rate on plateau
        if reduce_lr_on_plateau:
            lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_dice",
                patience=patience_lr_plat,
                mode="max",
                factor=factor_lr_plat,
                verbose=1,
                cooldown=5,
            )
            callbacks.append(lr_reduce_callback)

        # for tensorboard
        tb_callback = tf_utils.CustomTBCallback(
            output_path / "logs",
            update_freq="epoch",
            profile_batch=(2, 12),
            histogram_freq=1,
            embeddings_freq=0,
            write_grads=True,
            write_graph=write_graph,
            visualization_dataset=visualization_dataset,
            visualization_frequency=1,
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

    def _load_net(self):
        # This loads the keras network and the first checkpoint file
        self.model: tf.keras.Model = tf.keras.models.load_model(
            self.options["model_path"], compile=False, custom_objects={"Dice": Dice}
        )

        epoch = Path(self.options["model_path"]).name.split("-")[-1]
        self.variables["epoch"] = epoch

        if self.options["is_training"] is False:
            self.model.trainable = False
        self.options["rank"] = len(self.model.inputs[0].shape) - 2
        self.options["in_channels"] = self.model.inputs[0].shape.as_list()[-1]
        self.options["out_channels"] = self.model.outputs[0].shape.as_list()[-1]
        logger.info("Model was loaded")

    def get_hyperparameter_dict(self):
        """This function reads the hyperparameters from options and writes them to a dict of
        hyperparameters, which can then be read using tensorboard.

        Returns
        -------
        dict
            the hyperparameters as a dictionary
        """
        hyperparameters = {
            "dimension": self.options["rank"],
            "drop_out_rate": self.options["drop_out"][1],
            "regularize": self.options["regularize"][0],
            "regularizer": self.options["regularize"][1],
            "regularizer_param": self.options["regularize"][2],
            "kernel_dim": self.options["kernel_dims"][0],
            "dilation_rate_first": self.options["dilation_rate"][0],
        }
        if self.options["regularize"]:
            if isinstance(self.options["regularizer"], tf.keras.regularizers.L2):
                hyperparameters["L2"] = self.options["regularizer"].get_config()["l2"]
        # add filters
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
                    image_data, self.window_size
                )
                probability_patches = []
                # apply
                for x in image_data_patches:
                    probability_patches.append(self.model(x, training=False))
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
            useNearestNeighborExtrapolator=True,
        )

        # write resampled file
        name = Path(filename).name
        pred_path = Path(apply_path) / f"prediction-{name}-{version}{cfg.file_suffix}"
        sitk.WriteImage(predicted_label_orig, str(pred_path.resolve()))

        if cfg.write_probabilities:
            with open(Path(apply_path) / f"prediction-{name}-{version}.npy", "wb") as f:
                np.save(f, probability_map)

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
