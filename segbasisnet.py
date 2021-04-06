import logging
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from SegmentationNetworkBasis import tf_utils
from SegmentationNetworkBasis.NetworkBasis import loss
from SegmentationNetworkBasis.NetworkBasis.metric import Dice
from SegmentationNetworkBasis.NetworkBasis.network import Network

from . import config as cfg

#configure logger
logger = logging.getLogger(__name__)


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
                   l_r=0.001, optimizer='Adam', early_stopping=False, patience_es=10, reduce_lr_on_plateau=False, patience_lr_plat=5, factor_lr_plat=0.5,
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

        # check the iterator sizes
        iter_per_epoch = cfg.samples_per_volume * cfg.num_files // cfg.batch_size_train
        assert(iter_per_epoch > 0), 'Steps per epoch is zero, lower the batch size'
        iter_per_vald = cfg.samples_per_volume * cfg.number_of_vald // cfg.batch_size_valid
        assert(iter_per_vald > 0), 'Steps per epoch is zero for the validation, lower the batch size'

        # define callbacks
        callbacks = []

        # to save the model
        model_dir = output_path / 'models'
        if not model_dir.exists():
            model_dir.mkdir()

        # to save the best model
        cp_best_callback = tf_utils.Keep_Best_Model(
            filepath=model_dir / 'weights_best_{epoch:03d}-best{val_dice:1.3f}.hdf5',
            save_weights_only=True,
            verbose=0,
            save_freq='epoch',
            save_best_only=True,
            monitor='val_dice',
            mode='max'
        )
        callbacks.append(cp_best_callback)

        # backup and restore
        backup_dir = model_dir / 'backup'
        if not backup_dir.exists():
            backup_dir.mkdir()
        backup_callback = tf.keras.callbacks.experimental.BackupAndRestore(
            backup_dir=backup_dir,
        )
        callbacks.append(backup_callback)

        # early stopping
        if early_stopping:
            es_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_dice',
                patience=patience_es,
                mode='max',
                min_delta=0.005
            )
            callbacks.append(es_callback)

        # reduce learning rate on plateau
        if reduce_lr_on_plateau:
            lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_dice',
                patience=patience_lr_plat,
                mode='max',
                factor=factor_lr_plat,
                verbose=1,
                cooldown=5
            )
            callbacks.append(lr_reduce_callback)

        # for tensorboard
        tb_callback = tf_utils.Custom_TB_Callback(
            output_path / 'logs',
            update_freq='epoch',
            profile_batch=(2, 12),
            histogram_freq=1,
            write_graph=write_graph,
            visualization_dataset=visualization_dataset
        )
        callbacks.append(tb_callback)

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

        if 'CLUSTER' in os.environ:
            verbosity=2
        else:
            verbosity=1

        # do the training
        self.model.fit(
            x=training_dataset,
            epochs=epochs,
            verbose=verbosity,
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
            'kernel_dim' : self.options['kernel_dims'][0],
            'dilation_rate_first' : self.options['dilation_rate'][0]
        }
        if self.options['regularize']:
            if type(self.options['regularizer']) == tf.python.keras.regularizers.L2:
                hyperparameters['L2'] = self.options['regularizer'].get_config()['l2']
        # add filters
        for i, f in enumerate(self.options['n_filters']):
            hyperparameters[f'n_filters_{i}'] = f
        for o in ['use_bias', 'loss', 'name', 'batch_normalization', 'activation', 'use_cross_hair', 'res_connect', 'regularize', 'use_bias', 'skip_connect', 'n_blocks']:
            hyperparameters[o] = self.options[o]
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

        image_data = application_dataset(filename)

        # if 2D, run each layer as a batch, this should probably not run out of memory
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
                # if the window size exists, trigger the exception
                if hasattr(self, 'window_size'):
                    raise tf.errors.ResourceExhaustedError(None, None, 'trigger exception')
                probability_map = self.model(image_data, training=False)
            except tf.errors.ResourceExhaustedError:
                # try to reduce the patch size to a size that works
                if not hasattr(self, 'window_size'):
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
                        self.window_size[red] = int(np.ceil(self.window_size[red] / 16)) * 16
                        try:
                            test_data_shape = (1,) + tuple(self.window_size) + (image_data.shape[-1],)
                            test_data = np.random.rand(*test_data_shape)
                            self.model(test_data, training=False)
                        except tf.errors.ResourceExhaustedError:
                            logger.debug(f'Applying failed for window size {self.window_size} in step {i}.')
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
                            self.window_size[red] = int(np.ceil(self.window_size[red] / 16)) * 16
                            break
                # get windowed samples
                image_data_patches = application_dataset.get_windowed_test_sample(image_data, self.window_size)
                probability_patches = []
                # apply
                for x in image_data_patches:
                    probability_patches.append(self.model(x, training=False))
                probability_map = application_dataset.stitch_patches(probability_patches)

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
        sitk.WriteImage(predicted_label_orig, str(pred_path.resolve()))

        if cfg.write_probabilities:
            with open(Path(apply_path) / f'prediction-{name}-{version}.npy', 'wb') as f:
                np.save(f, probability_map)

        # write the preprocessed image
        proc_path = Path(apply_path) / f'sample-{name}-preprocessed{cfg.file_suffix}'
        sitk.WriteImage(orig_processed, str(proc_path.resolve()))

        # write the labels for the preprocessed image
        pred_res_path = Path(apply_path) / f'prediction-{name}-{version}-preprocessed{cfg.file_suffix}'
        sitk.WriteImage(sitk.Cast(predicted_label_img, sitk.sitkUInt8), str(pred_res_path.resolve()))

        return
