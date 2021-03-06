import logging
import os
from pathlib import Path
from time import time

import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from SegmentationNetworkBasis.NetworkBasis import image, loss
from SegmentationNetworkBasis.NetworkBasis.metric import Dice
from SegmentationNetworkBasis.NetworkBasis.network import Network

from . import config as cfg

#configure logger
logger = logging.getLogger(__name__)

# TODO: add validation summary using test_step

class SegBasisNet(Network):
    def __init__(self, loss, is_training=True, do_finetune=False, model_path="",
                 n_filters=[8, 16, 32, 64, 128], kernel_dims=3, n_convolutions=[2, 3, 2], drop_out=[False, 0.2],
                 regularize=[True, 'L2', 0.00001], do_batch_normalization=False, do_bias=True,
                 activation='relu', upscale='TRANS_CONV', downscale='MAX_POOL', res_connect=False, skip_connect=True,
                 cross_hair=False, **kwargs):

        # set tensorflow mixed precision policy #TODO: update for tf version 2.4
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)


        self.inputs = {}
        self.outputs = {}
        self.variables = {}
        self.options = {}
        self.options['is_training'] = is_training
        self.options['do_finetune'] = do_finetune
        self.options['regularize'] = regularize

        if not self.options['is_training'] or (self.options['is_training'] and self.options['do_finetune']):
            if model_path == "":
                raise ValueError('Model Path cannot be empty for Finetuning or Inference!')
        else:
            if model_path!= "":
                logger.warning("Caution: argument model_path is ignored in training!")

        self.options['model_path'] = model_path

        if self.options['is_training'] and not self.options['do_finetune']:
            self._set_up_inputs()
            self.options['n_filters'] = n_filters
            self.options['n_convolutions'] = n_convolutions
            self.options['drop_out'] = drop_out  # if [0] is True, dropout is added to the graph with keep_prob [2]
            self.options['regularizer'] = self._get_reg()
            self.options['batch_normalization'] = do_batch_normalization
            self.options['use_bias'] = do_bias
            if self.options['batch_normalization'] and self.options['use_bias']:
                logger.warning("Caution: do not use bias AND batch normalization!")
            self.options['activation'] = activation
            self.options['res_connect'] = res_connect
            self.options['skip_connect'] = skip_connect
            self.options['upscale'] = upscale  # #'BI_INTER' TRANS_CONV
            if self.options['upscale'] == 'UNPOOL_MAX_IND':
                self.variables['unpool_params'] = []
                self.options['downscale'] = 'MAX_POOL_ARGMAX'
                if not downscale == 'MAX_POOL_ARGMAX':
                    logger.warning("Caution: changed downscale to MAX_POOL_ARGMAX!")
            else:
                self.options['downscale'] = downscale
                if downscale == 'MAX_POOL_ARGMAX':
                    raise ValueError("MAX_POOL_ARGMAX has to be used with UNPOOL_MAX_IND!")
            self.options['padding'] = 'SAME'
            # if training build everything according to parameters
            self.options['in_channels'] = self.inputs['x'].shape.as_list()[-1]
            # self.options['out_channels'] is set elsewhere, but required
            self.options['rank'] = len(self.inputs['x'].shape) - 2  # input number of dimensions (without channels and batch size)
            self.options['use_cross_hair'] = cross_hair
            if self.options['rank'] == 2 and self.options['use_cross_hair']:
                logger.warning("Caution: cross_hair is ignored for 2D input!")
            self.options['dilation_rate'] = np.ones(self.options['rank'])  # input strides
            self.options['strides'] = np.ones(self.options['rank'], dtype=np.int32)  # outout strides
            self.options['kernel_dims'] = [kernel_dims] * self.options['rank']
            assert len(self.options['kernel_dims']) == self.options['rank']
            if self.options['skip_connect']:
                self.variables['feature_maps'] = []


            self.options['loss'] = loss
            self.outputs['loss'] = self._get_loss()
            self.model = self._build_model()
            self.model._name = self.get_name()
        else:
            # for finetuning load net from file
            self._load_net()
            self.options['loss'] = loss
            self.outputs['loss'] = self._get_loss()

        #write other kwags to options
        for key, value in kwargs.items():
            self.options[key] = value

    def _set_up_inputs(self):
        """setup the inputs. Inputs are taken from the config file.
        """
        self.inputs['x'] = tf.keras.Input(shape=cfg.train_input_shape, batch_size=cfg.batch_size_train, dtype=cfg.dtype)
        self.options['out_channels'] = cfg.num_classes_seg

    def _get_loss(self):
        '''!
        Returns loss depending on `self.options['loss']``.

        self.options['loss'] should be in {'DICE', 'TVE', 'GDL', 'CEL', 'WCEL'}.
        @returns @b loss : function
        '''
        if self.options['loss'] == 'DICE':
            return loss.dice_loss

        if self.options['loss'] == 'DICE-FNR':
            return loss.dice_with_fnr_loss

        elif self.options['loss'] == 'TVE':
            return loss.tversky_loss

        elif self.options['loss'] == 'GDL':
            return loss.generalized_dice_loss

        elif self.options['loss'] == 'GDL-FPNR':
            return loss.generalized_dice_with_fpr_fnr_loss

        elif self.options['loss'] == 'NDL':
            return loss.normalized_dice_loss

        elif self.options['loss'] == 'EDL':
            return loss.equalized_dice_loss

        elif self.options['loss'] == 'WDL':
            return loss.weighted_dice_loss

        elif self.options['loss'] == 'CEL':
            if self.options['out_channels'] > 2:
                return loss.categorical_cross_entropy_loss
            else:
                return loss.binary_cross_entropy_loss

        elif self.options['loss'] == 'ECEL':
            return loss.equalized_categorical_cross_entropy

        elif self.options['loss'] == 'NCEL':
            return loss.normalized_categorical_cross_entropy

        elif self.options['loss'] == 'WCEL':
            return loss.weighted_categorical_cross_entropy

        elif self.options['loss'] == 'ECEL-FNR':
            return loss.equalized_categorical_cross_entropy_with_fnr

        elif self.options['loss'] == 'WCEL-FPR':
                return loss.weighted_categorical_crossentropy_with_fpr_loss

        elif self.options['loss'] == 'GCEL':
                return loss.generalized_categorical_cross_entropy

        elif self.options['loss'] == 'CEL+DICE':
            return loss.categorical_cross_entropy_and_dice_loss

        else:
            raise ValueError(self.options['loss'], 'is not a supported loss function.')

    def _select_final_activation(self):
        # http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/
        if self.options['out_channels'] > 2 or self.options['loss'] in ['DICE', 'TVE', 'GDL']:
            # Dice, GDL and Tversky require SoftMax
            return 'softmax'
        elif self.options['out_channels'] == 2 and self.options['loss'] in ['CEL', 'WCEL', 'GCEL']:
            return 'sigmoid'
        else:
            raise ValueError(self.options['loss'],
                             'is not a supported loss function or cannot combined with ',
                             self.options['out_channels'], 'output channels.')

    def _run_train(self, logs_path, folder_name, training_dataset, validation_dataset, epochs,
                   l_r=0.001, optimizer='Adam', early_stopping=False, reduce_lr_on_plateau=False,
                   visualization_dataset=None, write_graph=True, **kwargs):
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
        reduce_lr_on_plateau : bool, optional
            If the learning rate should be reduced on a plateau, by default False
        visualization_dataset: SegBasisLoader, optional
            If provided, this dataset will be used to visualize the training results for debugging.
            Writing the images can take a bit, so only use this for debugging purposes.
        write_graph : bool, optional
            Controls if a graph should be written, can be used than only the first fold will
            get a graph, to prevent cluttering the output.
        """

        # set path
        output_path = Path(logs_path) / folder_name

        # do summary
        self.model.summary(print_fn=logger.info)

        # compile model
        self.model.compile(
            optimizer=self._get_optimizer(optimizer, l_r, 0),
            loss=self.outputs['loss'],
            metrics=[
                Dice(name='dice', num_classes=cfg.num_classes_seg),
                'acc',
                tf.keras.metrics.MeanIoU(num_classes=cfg.num_classes_seg)
            ]
        )

        # define callbacks
        callbacks = []
        iter_per_epoch = cfg.samples_per_volume * cfg.num_files // cfg.batch_size_train
        assert(iter_per_epoch > 0), 'Steps per epoch is zero, lower the batch size'
        iter_per_vald = cfg.samples_per_volume * cfg.number_of_vald // cfg.batch_size_valid
        assert(iter_per_vald > 0), 'Steps per epoch is zero for the validation, lower the batch size'
        # for tensorboard
        tb_callback = tf.keras.callbacks.TensorBoard(
            output_path / 'logs',
            update_freq='epoch',
            profile_batch=(2, 12),
            histogram_freq=1,
            write_graph=write_graph
        )
        callbacks.append(tb_callback)

        # to save the model
        model_dir = output_path / 'models'
        if not model_dir.exists():
            model_dir.mkdir()
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_dir / 'weights_{epoch:03d}.hdf5',
            save_weights_only=True,
            verbose=0,
            save_freq=int(10*iter_per_epoch)
        )
        callbacks.append(cp_callback)

        # to save the best model
        cp_best_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_dir / 'weights_best_{epoch:03d}-best{val_dice:1.3f}.hdf5',
            save_weights_only=True,
            verbose=0,
            save_freq='epoch',
            save_best_only=True,
            monitor='val_dice',
            mode='max'
        )
        callbacks.append(cp_best_callback)

        # early stopping
        if early_stopping:
            es_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_dice',
                patience=10,
                mode='max',
                min_delta=0.005
            )
            callbacks.append(es_callback)

        # reduce learning rate on plateau
        if reduce_lr_on_plateau:
            lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_dice',
                patience=5,
                mode='max',
                factor=0.5,
                verbose=1,
                cooldown=5
            )
            callbacks.append(lr_reduce_callback)

        # add callback vor visualization
        class Logger_Callback(tf.keras.callbacks.Callback):

            def __init__(self, visualization_dataset, writer):
                self.visualization_dataset = visualization_dataset
                self.writer = writer

            def on_epoch_end(self, epoch, logs):
                with self.writer.as_default():
                    # write learning rate
                    with tf.name_scope('learning_rate'):
                        tf.summary.scalar(f'learning_rate', self.model.optimizer.learning_rate, step=epoch)
                    # only write on every 5th epoch
                    if epoch %5 != 0:
                        return
                    if visualization_dataset != None:
                        # take one sample from the visualization dataset
                        for sample in self.visualization_dataset.take(1):
                            x, y = sample
                            probabilities = self.model(x)
                            write_images(x, y, probabilities, step=epoch)
                        return

        # log additional data (using the tensorboard writer)
        writer = tf.summary.create_file_writer(str(output_path / 'logs' / 'lr_and_images'))
        image_visualization_and_lr_callback = Logger_Callback(visualization_dataset, writer)
        callbacks.append(image_visualization_and_lr_callback)   

        # callback for hyperparameters
        hparams = self.get_hyperparameter_dict()
        # set additional paramters
        hparams['folder_name'] = folder_name
        hp_callback = hp.KerasCallback(str(output_path / 'logs' / 'train'), hparams)
        callbacks.append(hp_callback)

        # callback to write csv data
        csv_callback = tf.keras.callbacks.CSVLogger(
            filename = output_path / 'training.csv',
            separator=';'
        )
        callbacks.append(csv_callback)

        # do the training
        self.model.fit(
            x=training_dataset,
            epochs=epochs,
            verbose=1,
            validation_data=validation_dataset,
            validation_freq=1,
            steps_per_epoch=iter_per_epoch,
            validation_steps=iter_per_vald,
            callbacks=callbacks
        )
        print('Saving the final model.')
        # save the model once (only weights are saved afterwards)
        self.model.save(model_dir / 'model-final', save_format='tf')
        print('Saving finished.')

    def _load_net(self):
        # This loads the keras network and the first checkpoint file
        self.model = tf.keras.models.load_model(
            self.options['model_path'],
            compile=False,
            custom_objects={'Dice' : Dice}
        )

        self.variables['epoch'] = 'final'

        if self.options['is_training'] == False:
            self.model.trainable = False
        self.options['rank'] = len(self.model.inputs[0].shape) - 2
        self.options['in_channels'] = self.model.inputs[0].shape.as_list()[-1]
        self.options['out_channels'] = self.model.outputs[0].shape.as_list()[-1]
        logger.info('Model was loaded')

    def get_hyperparameter_dict(self):
        """This function reads the hyperparameters from options and writes them to a dict of
        hyperparameters, which can then be read using tensorboard.

        Returns
        -------
        dict
            the hyperparameters as a dictionary
        """
        hyperparameters = {
            'dimension' : self.options['rank'],
            'drop_out_rate' : self.options['drop_out'][1],
            'regularize' : self.options['regularize'][0],
            'regularizer' : self.options['regularize'][1],
            'regularizer_param' : self.options['regularize'][2],
            'kernel_dim' : self.options['kernel_dims'][0]
        }
        for o in ['use_bias', 'loss', 'name', 'batch_normalization', 'activation', 'use_cross_hair']:
            hyperparameters[o] = self.options[o]
        # check the types
        for key, value in hyperparameters.items():
            if type(value) in [list]:
                hyperparameters[key] = str(value)
        return hyperparameters

    # TODO: simplify and document
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

        image_data = application_dataset(filename)

        # if 2D, run each layer as a batch
        if self.options['rank'] == 2:
            predictions = []
            for x in np.expand_dims(image_data, axis=1):
                p = self.model(x, training=False)
                predictions.append(p)
            # concatenate
            probability_map = np.concatenate(predictions)
        # otherwise, just run it
        else:
            try:
                probability_map = self.model(image_data, training=False)
            except tf.errors.ResourceExhaustedError:
                logger.info(f'{filename} cannot be processed as a complete image, dividing into slices.')
                # if running out of memory, turn into sub-problems
                assert application_dataset.training_shape[0] % 2 == 0, 'Training Shape z-extent should be divisible by 2'
                z_half = application_dataset.training_shape[0] // 2
                # always use the two central slices
                indices = np.arange(z_half, image_data.shape[1] - z_half, 2, dtype=int)
                # create a new probability map
                map_shape = image_data.shape[:-1] + (cfg.num_classes_seg,)
                probability_map = np.zeros(map_shape)
                for i in indices:
                    result = self.model(image_data[:,i-z_half:i+z_half], training=False)
                    # take the central two slices
                    probability_map[:,i-1:i+1] = result[:,z_half-1:z_half+1]

        # remove padding
        probability_map = application_dataset.remove_padding(probability_map)

        # remove the batch dimension
        if self.options['rank'] == 3:
            probability_map = probability_map[0]
        
        # get the labels
        predicted_labels = np.argmax(probability_map, -1)

        # get the processed image
        orig_processed = application_dataset.get_processed_image(filename)

        predicted_label_img = sitk.GetImageFromArray(predicted_labels)
        # copy info
        predicted_label_img.CopyInformation(orig_processed)

        # get the original image
        original_image = application_dataset.get_original_image(filename)

        # resample to the original file
        predicted_label_orig = sitk.Resample(
            image1=predicted_label_img,
            referenceImage=original_image,
            interpolator=sitk.sitkNearestNeighbor,
            outputPixelType =sitk.sitkUInt8,
            useNearestNeighborExtrapolator=True
        )

        # write resampled file
        name = Path(filename).name
        pred_path = Path(apply_path) / f'prediction-{name}-{version}{cfg.file_suffix}'
        sitk.WriteImage(predicted_label_orig, str(pred_path.absolute()))

        if cfg.write_probabilities:
            with open(Path(apply_path) / f'prediction-{name}-{version}.npy', 'wb') as f:
                np.save(f, probability_map)

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
            image_fc = tf.expand_dims(tf.cast((tf.clip_by_value(x, -1, 1) + 1) * 255 / 2, tf.uint8), axis=-1)
            tf.summary.image('train_img', image_fc, step, max_image_output)
        else:
            for c in range(in_channels):
                image = tf.expand_dims(tf.cast((tf.clip_by_value(x[:, :, :, c], -1, 1) + 1) * 255 / 2, tf.uint8), axis=-1)
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