"""This module contains some tensorflow Callbacks, which can be used for nicer
logging and to only save the best weights, which does save disk space.
"""

import logging
from pathlib import Path
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras import callbacks

# configure logger
logger = logging.getLogger(__name__)

# some callbacks do only have one public method.
# pylint: disable=too-few-public-methods


def get_optimizer(
    optimizer: str, l_r: float, clipvalue: Optional[float] = None
) -> tf.optimizers.Optimizer:
    """Get the optimizer according to the name

    Parameters
    ----------
    optimizer : str
        The name of the optimizer
    l_r : float
        The learning rate
    clipvalue : float, optional
        At which value the gradients should be clipped, by default None

    Returns
    -------
    tf.optimizers.Optimizer
        The optimizer

    Raises
    ------
    ValueError
        If the name is unknown
    """
    if optimizer == "Adam":
        return tf.optimizers.Adam(learning_rate=l_r, epsilon=1e-3, clipvalue=clipvalue)
    elif optimizer == "Momentum":
        mom = 0.9
        learning_rate = tf.optimizers.schedules.ExponentialDecay(
            l_r, 6000, 0.96, staircase=True
        )
        return tf.optimizers.SGD(learning_rate, momentum=mom, clipvalue=clipvalue)
    elif optimizer == "Adadelta":
        return tf.optimizers.Adadelta(learning_rate=l_r, clipvalue=clipvalue)
    elif optimizer == "SGD":
        return tf.optimizers.SGD(learning_rate=l_r, clipvalue=None)
    elif optimizer == "RMSprop":
        return tf.optimizers.RMSprop(learning_rate=l_r, clipvalue=None)
    elif optimizer == "Adamax":
        return tf.optimizers.Adamax(learning_rate=l_r, epsilon=1e-3, clipvalue=None)
    else:
        raise ValueError(f"Optimizer {optimizer} unknown.")


class KeepBestModel(callbacks.ModelCheckpoint):
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
        If only the best model(s) should be saved, by default True
    decay : float, optional
        If there should be an exponential moving average for the value with the
        specified rate.
    """

    def __init__(self, filepath, max_keep=3, save_best_only=True, decay=None, **kwargs):
        super().__init__(filepath, save_best_only=save_best_only, **kwargs)
        # maximum number of checkpoints to keep
        self.max_keep = max_keep
        self.best_checkpoints = {}
        self.decay = decay
        if decay is not None:
            self.previous_val = 0

    def on_epoch_end(self, epoch, logs=None):
        """On epoch end, save the checkpoint if it was better than max_keep and
        delete the worst one.
        """
        # get the value
        if self.decay is None:
            val = logs[self.monitor]
        # with moving average if specified
        else:
            val = self.decay * self.previous_val + (1 - self.decay) * logs[self.monitor]
            self.previous_val = val
            logger.info("Monitored values %s is %5f", self.monitor, logs[self.monitor])

        # see if it was better than any checkpoint (or all should be saved)
        save = not self.save_best_only
        for best_val in self.best_checkpoints:
            # see if it is better
            if self.monitor_op(val, best_val):
                save = True
        # see if there are less than max_keep checkpoints
        if len(self.best_checkpoints) < self.max_keep:
            save = True

        # return if it was not better
        if not save:
            logger.info("Value %5f for %s did not improve", val, self.monitor)
            return

        logger.info(
            "Value %5f for %s did improve and the weights will be saved.",
            val,
            self.monitor,
        )

        # save it
        self._save_model(epoch, logs)
        # write it to the dictionary
        self.best_checkpoints[val] = self._get_file_path(epoch, logs)

        # see if there are checkpoints than should be removed
        if len(self.best_checkpoints) > self.max_keep and self.save_best_only:
            worst_value = None
            worst_checkpoint = None
            # iterate over all checkpoints
            for chk_val, chk_file in self.best_checkpoints.items():
                # remember the worst value
                if worst_value is None or self.monitor_op(worst_value, chk_val):
                    worst_value = chk_val
                    worst_checkpoint = chk_file
            # remove from list
            logger.info("Worst file %s will be deleted.", worst_checkpoint)
            self.best_checkpoints.pop(worst_value)
            try:
                Path(worst_checkpoint).unlink()
            except FileNotFoundError:
                logger.info("The file %s was not found", worst_checkpoint)

    def _save_model(self, epoch, logs):
        filename = self._get_file_path(epoch, logs)
        if self.save_weights_only:
            self.model.save_weights(filename, overwrite=True, options=self._options)
        else:
            self.model.save(filename, overwrite=True, options=self._options)
        if not Path(filename).exists():
            raise FileNotFoundError(f"The saved model {filename} was not found.")

    def _get_file_path(self, epoch, logs):
        return self.filepath.format(epoch=epoch + 1, **logs)


class FinetuneLayers(callbacks.Callback):
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
        If batchnorm layers should be trainable, not recommended for finetuning, by default True
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


class CustomTBCallback(callbacks.TensorBoard):
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
    write_grads : bool
        If gradients should be written as images, by default False
    write_labels : bool
        If labels should be written as images, by default False
    ignore : List
        The indices of the output, that should be ignored
    **kwargs
        All other arguments will be passed on to tf.keras.callbacks.TensorBoard.
    """

    def __init__(
        self,
        log_dir,
        visualization_dataset=None,
        visualization_frequency=5,
        write_grads=False,
        write_labels=False,
        ignore=None,
        **kwargs,
    ):
        super().__init__(log_dir=log_dir, **kwargs)
        self.visualization_dataset = visualization_dataset
        self.visualization_frequency = visualization_frequency
        self.write_grads = write_grads
        self.write_labels = write_labels
        if ignore is None:
            ignore = []
        self.ignore = ignore

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
                l_r = self.model.optimizer.learning_rate
                if callable(l_r):
                    l_r = l_r(self.params["steps"] * epoch)
                tf.summary.scalar("learning_rate", l_r, step=epoch)
            # only write on every epoch divisible by visualization_frequency
            if epoch % self.visualization_frequency != 0:
                return
            # write images
            if self.visualization_dataset is not None:
                # take one sample from the visualization dataset
                for sample in self.visualization_dataset.take(1):
                    x, y = sample
                    y_pred = self.model(x)
                    write_images(
                        x=x,
                        y=y,
                        y_pred=y_pred,
                        step=epoch,
                        num_segmentations=self.write_labels,
                        ignore=self.ignore,
                    )
            # write gradients
            if self.write_grads:
                if self.visualization_dataset is None:
                    raise ValueError(
                        "Visualization Dataset should be provided for gradients."
                    )
                try:
                    gradients = self.get_gradients(self.visualization_dataset)
                except tf.errors.ResourceExhaustedError as err:
                    tf.print("OOM Error when calculating gradients, skipped and disabled")
                    self.write_grads = False
                    logger.exception(err)
                    return
                # write gradients
                for weights, grads in zip(self.model.trainable_weights, gradients):
                    if grads is None:
                        continue
                    tf.summary.histogram(
                        weights.name.replace(":", "_") + "_grads",
                        data=grads,
                        step=epoch,
                    )


def write_images(x, y, y_pred, step: int, num_segmentations=0, ignore=None):
    """Write images for the summary. If 3D data is provided, the central slice
    is used. All channels are written, the labels are written and the
    probabilities. If additional images are provided after the labels, they will
    be written as well.

    Parameters
    ----------
    x : tf.Tensor|Tuple[tf.Tensor]
        The input images as Tuple of Tensors
    y : tf.Tensor|Tuple[tf.Tensor]
        The ground truth
    y_pred : tf.Tensor|Tuple[tf.Tensor]
        The output of the network
    step : int
        Step number used for slider in tensorboard
    num_segmentations : int, optional
        The number of segmentation labels in the results, by default 0
    ignore : List, optional
        Which fields should be ignored in y_pred, by default None
    """

    # if it is not a Iterable, make it one
    if not isinstance(x, tuple) and not isinstance(x, list):
        x = (x,)
    if not isinstance(y, tuple) and not isinstance(y, list):
        y = (y,)
    if not isinstance(y_pred, tuple) and not isinstance(y_pred, list):
        y_pred = (y_pred,)
    if ignore is not None:
        y_pred = tuple(y_p for n, y_p in enumerate(y_pred) if n not in ignore)

    dimension = len(x[0].shape) - 2  # subtract one dimension for batches and channels
    with tf.name_scope("Input"):
        for num, img in enumerate(x):
            in_channels = img.shape[-1]
            max_image_output = 1
            name = f"train_img_{num}"
            # take central slice of 3D data
            if dimension == 3:
                dim_z = img.shape[1]
                img = img[:, dim_z // 2, :, :]
            if in_channels == 1:
                image_fc = convert_float_to_image(img)
                if image_fc.ndim == 5:
                    image_fc = image_fc[..., 0]
                tf.summary.image(name, image_fc, step, max_image_output)
            else:
                for cls in range(in_channels):
                    image = convert_float_to_image(img[:, :, :, cls])
                    if cls == 0:
                        image_fc = image
                    tf.summary.image(name + "_c" + str(cls), image, step, max_image_output)

    # write all output images
    with tf.name_scope("Prediction"):
        for img_num, pred in enumerate(y_pred):
            # only show images
            if len(pred.shape) < 4:
                continue
            dimension = len(pred.shape) - 2
            n_channels = pred.shape[-1]
            if dimension == 3:
                dim_z = pred.shape[1]
                pred = pred[:, dim_z // 2, :, :]

            for cls in range(n_channels):
                img = convert_float_to_image(pred[:, :, :, cls])
                tf.summary.image(
                    f"pred_nr_{img_num}_channel_{cls}",
                    img,
                    step,
                    max_image_output,
                )

    if num_segmentations > 0:
        for img_num, probabilities in enumerate(y_pred[:num_segmentations]):
            predictions = tf.argmax(probabilities, -1)
            n_channels = probabilities.shape[-1]
            with tf.name_scope("Segmentation Results"):
                if dimension == 3:
                    y = y[:, dim_z // 2, :, :]
                    probabilities = probabilities[:, dim_z // 2, :, :]
                    predictions = predictions[:, dim_z // 2, :, :]

                label = tf.expand_dims(
                    tf.cast(tf.argmax(y, -1) * (255 // (n_channels - 1)), tf.uint8),
                    axis=-1,
                )
                tf.summary.image(f"train_seg_lbl_{img_num}", label, step, max_image_output)
                pred = tf.expand_dims(
                    tf.cast(predictions * (255 // (n_channels - 1)), tf.uint8), axis=-1
                )
            with tf.name_scope(
                "Combined_predictions (prediction in red, label in green, both in yellow)"
            ):
                tf.summary.image(f"train_seg_pred_{img_num}", pred, step, max_image_output)
                # set to first channel where both labels are zero
                mask = tf.cast(tf.math.logical_and(pred == 0, label == 0), tf.uint8)
                # set those values to the mask
                label += image_fc * mask
                pred += image_fc * mask
                # set the opposite values of the image to zero
                image_fc -= image_fc * (1 - mask)
                combined = tf.concat([pred, label, image_fc], -1)
                tf.summary.image(
                    f"train_seg_combined_{img_num}", combined, step, max_image_output
                )

            with tf.name_scope("04_Class_Labels"):
                if n_channels == 2:
                    pass
                else:
                    for cls in range(n_channels):
                        tf.summary.image(
                            f"train_seg_lbl_{img_num}" + str(cls),
                            tf.expand_dims(
                                tf.cast(y[:, :, :, cls] * 255, tf.uint8), axis=-1
                            ),
                            step,
                            max_image_output,
                        )


def convert_float_to_image(image: tf.Tensor) -> tf.Tensor:
    """Convert a float tensor to a grayscale image with values between 0 and 255.
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
