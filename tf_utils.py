"""This module contains some tensorflow Callbacks, which can be used for nicer
logging and to only save the best weights, which does save disk space.
"""

from pathlib import Path

import tensorflow as tf
from . import config as cfg

class Keep_Best_Model(tf.keras.callbacks.ModelCheckpoint):
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
        assert save_best_only, 'has to be true (only there for compability.'
        super().__init__(filepath, save_best_only=True, **kwargs)
        # maximum number of checkpoints to keep
        self.max_keep=max_keep
        self._best_checkpoints = {}

    def on_epoch_end(self, epoch, logs):
        super().on_epoch_end(epoch, logs=logs)
        filename = self._get_file_path(epoch, logs)
        # if the file was not saved, return
        if not self._checkpoint_exists(filename):
            return
        # get the value
        val = logs[self.monitor]
        # save it
        self._best_checkpoints[val] = filename
        # see if there are more checkpoints than should be kept
        if len(self._best_checkpoints) > self.max_keep:
            worst_value = None
            worst_checkpoint = None
            # iterate over all checkpoints
            for chk_val, chk_file in self._best_checkpoints.items():
                # remember the worst value
                if worst_value is None or self.monitor_op(worst_value, chk_val):
                    worst_value = chk_val
                    worst_checkpoint = chk_file
            # remove from list
            self._best_checkpoints.pop(worst_value)
            if self.save_weights_only:
                Path(worst_checkpoint).unlink()


class Custom_TB_Callback(tf.keras.callbacks.TensorBoard):
    """Extended TensorBoard callback, it will also always log the learning rate
    and write images of the segmentation if a visualization dataset is provided.

    Parameters
    ----------
    log_dir : str
        The location of the log
    visualization_dataset : tf.data.Dataset
        The dataset for visualization. This will only be executed every 5 epochs,
        because it is computationally expensive. If None, there will be no
        visualization, by default None
    **kwargs
        All other arguments will be passed on to tf.keras.callbacks.TensorBoard.
    """

    def __init__(self, log_dir, visualization_dataset=None, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
        self.visualization_dataset = visualization_dataset

    def on_epoch_end(self, epoch, logs):
        super().on_epoch_end(epoch, logs=logs)
        with self._train_writer.as_default():
            # write learning rate
            with tf.name_scope('learning_rate'):
                tf.summary.scalar(f'learning_rate', self.model.optimizer.learning_rate, step=epoch)
            # only write on every 5th epoch
            if epoch %5 != 0:
                return
            if self.visualization_dataset != None:
                # take one sample from the visualization dataset
                for sample in self.visualization_dataset.take(1):
                    x, y = sample
                    probabilities = self.model(x)
                    write_images(x, y, probabilities, step=epoch)
        return


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
    dimension = len(x.shape) - 2 # substract one dimension for batches and channels
    max_image_output=1

    # take central slice of 3D data
    if dimension == 3:
        dim_z = x.shape[1]
        x = x[:, dim_z // 2, :, :]
        y = y[:, dim_z // 2, :, :]
        probabilities = probabilities[:, dim_z // 2, :, :]

    predictions = tf.argmax(probabilities, -1)


    with tf.name_scope('01_Input_and_Predictions'):
        if in_channels == 1:
            image_fc = convert_float_to_image(x)
            tf.summary.image('train_img', image_fc, step, max_image_output)
        else:
            for c in range(in_channels):
                image = convert_float_to_image(x[:, :, :, c])
                if c == 0:
                    image_fc = image
                tf.summary.image('train_img_c'+str(c), image, step, max_image_output)

        label = tf.expand_dims(tf.cast(tf.argmax(y, -1)
                * (255 // (cfg.num_classes_seg - 1)), tf.uint8), axis=-1)
        tf.summary.image('train_seg_lbl', label, step, max_image_output)
        pred = tf.expand_dims(tf.cast(predictions
                * (255 // (cfg.num_classes_seg - 1)), tf.uint8), axis=-1)
        tf.summary.image('train_seg_pred', pred, step, max_image_output)


    with tf.name_scope('02_Combined_predictions (prediction in red, label in green, both in yellow)'):
        # set to first channel where both labels are zero
        mask = tf.cast(tf.math.logical_and(pred == 0, label == 0), tf.uint8)
        # set those values to the mask
        label += image_fc*mask
        pred += image_fc*mask
        # set the opposite values of the image to zero
        image_fc -= image_fc*(1-mask)
        combined = tf.concat([pred, label, image_fc], -1)
        tf.summary.image('train_seg_combined', combined, step, max_image_output)


    with tf.name_scope('03_Probabilities'):
        if dimension == 2:
            for c in range(cfg.num_classes_seg):
                tf.summary.image('train_seg_prob_' + str(c), tf.expand_dims(tf.cast(probabilities[:, :, :, c]
                    * 255, tf.uint8), axis=-1), step, max_image_output)


    with tf.name_scope('04_Class_Labels'):
        if cfg.num_classes_seg == 2:
            pass
        else:
            for c in range(cfg.num_classes_seg):
                tf.summary.image('train_seg_lbl' + str(c), tf.expand_dims(tf.cast(y[:, :, :, c]
                    * 255, tf.uint8), axis=-1), step, max_image_output)

    return

def convert_float_to_image(image:tf.Tensor)->tf.Tensor:
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