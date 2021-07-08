"""This module contains some tensorflow Callbacks, which can be used for nicer
logging and to only save the best weights, which does save disk space.
"""

import logging
from pathlib import Path
from typing import List

import tensorflow as tf
from . import config as cfg

# configure logger
logger = logging.getLogger(__name__)

# some callbacks do only have one public method.
# pylint: disable=too-few-public-methods


class KeepBestModel(tf.keras.callbacks.ModelCheckpoint):
    """This extends the tf.keras.callbacks.ModelCheckpoint class to delete the
    worst model once more than max_keep models are saved. This can help to
    reduce the amount of storage needed for the training.

    Parameters
    ----------
    filepath : str
        Where the models should be saved
    max_keep : int, optional
        The maximum amount of models/weights to keep, by default 3
    save_best_only : bool, optional
        Always true, is there only for compability, by default True
    """

    def __init__(self, filepath, max_keep=3, save_best_only=True, **kwargs):
        assert save_best_only, "has to be true (only there for compability."
        super().__init__(filepath, save_best_only=True, **kwargs)
        # maximum number of checkpoints to keep
        self.max_keep = max_keep
        self.best_checkpoints = {}

    def on_epoch_end(self, epoch, logs=None):
        """On epoch end, save the checkpoint if it was better than max_keep and
        delete the worst one.
        """
        super().on_epoch_end(epoch, logs=logs)
        filename = self._get_file_path(epoch, logs)
        # if the file was not saved, return
        if not self._checkpoint_exists(filename):
            return
        # get the value
        val = logs[self.monitor]
        # save it
        self.best_checkpoints[val] = filename
        # see if there are more checkpoints than should be kept
        if len(self.best_checkpoints) > self.max_keep:
            worst_value = None
            worst_checkpoint = None
            # iterate over all checkpoints
            for chk_val, chk_file in self.best_checkpoints.items():
                # remember the worst value
                if worst_value is None or self.monitor_op(worst_value, chk_val):
                    worst_value = chk_val
                    worst_checkpoint = chk_file
            # remove from list
            self.best_checkpoints.pop(worst_value)
            if self.save_weights_only:
                Path(worst_checkpoint).unlink()


class FinetuneLayers(tf.keras.callbacks.Callback):
    """For finetuning, this callback will enable the training of certain layers
    at a selected epoch.

    Parameters
    ----------
    to_activate : List, optional
        Which layers should be finetuned. This can either be a list of names or all,
        which enables training on all layers (besides batchnorm layers if disabled), by default None
    epoch : int, optional
        At which epoch fine-tuning should be enabled, if None, no finetuning will be done, by default 10
    train_bn : bool, optional
        If batchnorm layers should be trainable, not recommended for finetuning, by default False
    learning_rate : float, optional
        If not None, this rate will be set after enabling the finetuning, by default None
    """

    def __init__(self, to_activate=None, epoch=10, train_bn=True, learning_rate=None):

        self.to_activate = to_activate
        self.epoch = epoch
        self.train_bn = train_bn
        self.learning_rate = learning_rate
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        """Enable the training at the begin of the selected epoch."""
        if epoch == self.epoch:
            if self.to_activate == "all":
                for layer in self.model.layers:
                    # do not add batch norm layers
                    if not self.train_bn and isinstance(
                        layer, tf.keras.layers.BatchNormalization
                    ):
                        continue
                    if not layer.trainable:
                        layer.trainable = True
                        logger.debug("Made layer %s trainable", layer.name)
            elif isinstance(self.to_activate, list):
                for l_name in self.to_activate:
                    self.model.get_layer(l_name).trainable = True
                    logger.debug("Made layer %s trainable", l_name)
            if self.learning_rate is not None:
                tf.keras.backend.set_value(self.model.optimizer.lr, self.learning_rate)
                logger.info("Learning rate changed to %f", self.learning_rate)

        return super().on_epoch_begin(epoch, logs=logs)


class CustomTBCallback(tf.keras.callbacks.TensorBoard):
    """Extended TensorBoard callback, it will also always log the learning rate
    and write images of the segmentation if a visualization dataset is provided.
    The visualization takes about 10 seconds, so for a lot of epochs, a frequency
    of more than 1 should be used.

    Parameters
    ----------
    log_dir : str
        The location of the log
    visualization_dataset : tf.data.Dataset
        The dataset for visualization. If None, there will be no
        visualization, by default None
    visualization_frequency : float
        How often images and gradients should be written
    **kwargs
        All other arguments will be passed on to tf.keras.callbacks.TensorBoard.
    """

    def __init__(
        self,
        log_dir,
        visualization_dataset=None,
        visualization_frequency=5,
        write_grads=True,
        **kwargs,
    ):
        super().__init__(log_dir=log_dir, **kwargs)
        self.visualization_dataset = visualization_dataset
        self.visualization_frequency = visualization_frequency
        self.write_grads = write_grads

    @tf.function
    def get_gradients(self, dataset: tf.data.Dataset) -> List[tf.Tensor]:
        """Get the gradients for the model and the given dataset.

        Parameters
        ----------
        dataset : tf.data.Dataset
            The dataset to use. Only one element is used.

        Returns
        -------
        List[tf.Tensor]
            The gradients with respect to the loss. The list has the same members
            and dimensions as self.model.trainable_weights
        """
        sample = next(iter(dataset))
        x, y = sample
        with tf.GradientTape() as tape:
            # predict it
            probabilities = self.model(x)
            # get the loss
            loss = self.model.compiled_loss(y_true=y, y_pred=probabilities)
            # do backpropagation
            gradients = tape.gradient(loss, self.model.trainable_weights)
        return gradients

    def on_epoch_end(self, epoch, logs=None):
        """Write metrics to tensorboard at the end of the epoch."""
        super().on_epoch_end(epoch, logs=logs)
        with self._train_writer.as_default():
            # write learning rate
            with tf.name_scope("learning_rate"):
                tf.summary.scalar(
                    "learning_rate", self.model.optimizer.learning_rate, step=epoch
                )
            # only write on every epoch divisible by visualization_frequency
            if epoch % self.visualization_frequency != 0:
                return
            # write gradients
            if self.write_grads:
                if self.visualization_dataset is None:
                    raise ValueError(
                        "Visualization Dataset should be provided for gradients."
                    )
                gradients = self.get_gradients(self.visualization_dataset)
                # write gradients
                for weights, grads in zip(self.model.trainable_weights, gradients):
                    tf.summary.histogram(
                        weights.name.replace(":", "_") + "_grads", data=grads, step=epoch
                    )
            if self.visualization_dataset is not None:
                # take one sample from the visualization dataset
                for sample in self.visualization_dataset.take(1):
                    x, y = sample
                    probabilities = self.model(x)
                    write_images(x, y, probabilities, step=epoch)


def write_images(x, y, probabilities, step):
    """Write images for the summary. If 3D data is provided, the central slice
    is used. All channels are written, the labels are written and the
    probabilites.

    Parameters
    ----------
    x : tf.Tensor
        The input images as Tensor
    y : tf.Tensor
        The input labels
    probabilities : tf.Tensor
        The output of the network as probabilites (one per class)
    step : int
        Step number used for slider in tensorboard
    """

    in_channels = x.shape[-1]
    dimension = len(x.shape) - 2  # substract one dimension for batches and channels
    max_image_output = 1

    # take central slice of 3D data
    if dimension == 3:
        dim_z = x.shape[1]
        x = x[:, dim_z // 2, :, :]
        y = y[:, dim_z // 2, :, :]
        probabilities = probabilities[:, dim_z // 2, :, :]

    predictions = tf.argmax(probabilities, -1)

    with tf.name_scope("01_Input_and_Predictions"):
        if in_channels == 1:
            image_fc = convert_float_to_image(x)
            tf.summary.image("train_img", image_fc, step, max_image_output)
        else:
            for cls in range(in_channels):
                image = convert_float_to_image(x[:, :, :, cls])
                if cls == 0:
                    image_fc = image
                tf.summary.image("train_img_c" + str(cls), image, step, max_image_output)

        label = tf.expand_dims(
            tf.cast(tf.argmax(y, -1) * (255 // (cfg.num_classes_seg - 1)), tf.uint8),
            axis=-1,
        )
        tf.summary.image("train_seg_lbl", label, step, max_image_output)
        pred = tf.expand_dims(
            tf.cast(predictions * (255 // (cfg.num_classes_seg - 1)), tf.uint8), axis=-1
        )
        tf.summary.image("train_seg_pred", pred, step, max_image_output)

    with tf.name_scope(
        "02_Combined_predictions (prediction in red, label in green, both in yellow)"
    ):
        # set to first channel where both labels are zero
        mask = tf.cast(tf.math.logical_and(pred == 0, label == 0), tf.uint8)
        # set those values to the mask
        label += image_fc * mask
        pred += image_fc * mask
        # set the opposite values of the image to zero
        image_fc -= image_fc * (1 - mask)
        combined = tf.concat([pred, label, image_fc], -1)
        tf.summary.image("train_seg_combined", combined, step, max_image_output)

    with tf.name_scope("03_Probabilities"):
        if dimension == 2:
            for cls in range(cfg.num_classes_seg):
                tf.summary.image(
                    "train_seg_prob_" + str(cls),
                    tf.expand_dims(
                        tf.cast(probabilities[:, :, :, cls] * 255, tf.uint8), axis=-1
                    ),
                    step,
                    max_image_output,
                )

    with tf.name_scope("04_Class_Labels"):
        if cfg.num_classes_seg == 2:
            pass
        else:
            for cls in range(cfg.num_classes_seg):
                tf.summary.image(
                    "train_seg_lbl" + str(cls),
                    tf.expand_dims(tf.cast(y[:, :, :, cls] * 255, tf.uint8), axis=-1),
                    step,
                    max_image_output,
                )


def convert_float_to_image(image: tf.Tensor) -> tf.Tensor:
    """Convert a float tensor to a greyscale image with values between 0 and 255.
    This is done by setting the minimum to 0 and the maximum to 255. It is assumed
    that outliers were already removed.

    Parameters
    ----------
    image : tf.Tensor
        The tensor to convert

    Returns
    -------
    tf.Tensor
        The image
    """
    # get the extreme values
    minimum = tf.math.reduce_min(image)
    maximum = tf.math.reduce_max(image)
    # rescale to a range between 0 and 255
    image_rescaled = (image - minimum) / (maximum - minimum) * 255
    # cast to int and add channel dimension
    return tf.expand_dims(tf.cast(image_rescaled, tf.uint8), axis=-1)
