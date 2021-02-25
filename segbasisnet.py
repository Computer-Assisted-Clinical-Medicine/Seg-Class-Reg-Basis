import logging
import os
from pathlib import Path
from time import time

import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import tensorflow.profiler.experimental as profiler
from tqdm import tqdm

from . import config as cfg
from .NetworkBasis import image, loss, metric
from .NetworkBasis.network import Network

#configure logger
logger = logging.getLogger(__name__)

class SegBasisNet(Network):

    def _set_up_inputs(self):
        self.inputs['x'] = tf.keras.Input(shape=cfg.train_input_shape, batch_size=cfg.batch_size_train, dtype=cfg.dtype)
        self.options['out_channels'] = cfg.num_classes_seg

    @staticmethod
    def _get_test_size():
        return cfg.test_data_shape

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

    def _run_apply(self, version, apply_path, application_dataset, filename):

        if not os.path.exists(apply_path):
            os.makedirs(apply_path)

        predictions = []
        start_time = time()

        for x in application_dataset:
            prediction = self.model(x, training=False)
            predictions.append(prediction)

        if self.options['rank'] == 3:
            m = len(predictions)
            extend_z = cfg.test_label_shape[0] * np.int(np.ceil(m/2))

            probability_map = np.zeros((extend_z, *cfg.test_label_shape[1:3], self.options['out_channels']))
            # weight_map = np.zeros((extend_z, *cfg.test_label_shape[1:3], self.options['out_channels']))
            basic_weights = np.ones(predictions[0].shape, dtype=np.float)
            part_length = cfg.test_label_shape[0] // 2
            weight_factors = np.linspace(0, 1, part_length)
            weight_factors = np.concatenate([weight_factors, np.flip(weight_factors)])
            for i in range(cfg.test_label_shape[0]):
                basic_weights[:, i] = weight_factors[i]

            for p in range(0, m):
                if p == 0:
                    # The first half of the first prediction gets weighted with 1, the rest with 0.5
                    weights = basic_weights.copy()
                    weights[:, :part_length] = 1.0
                elif p == m-1:
                    # The second half of the last prediction gets weighted with 1, the rest with 0.5
                    weights = basic_weights.copy()
                    weights[:, part_length:] = 1.0
                else:
                    weights = basic_weights.copy()

                probability_map[p * part_length: p * part_length + cfg.test_label_shape[0]] += np.squeeze(np.multiply(weights, predictions[p]))
                # weight_map[p * part_length: p * part_length + cfg.test_label_shape[0]] += np.squeeze(weights)

                # import matplotlib.pyplot as plt
                # plt.imshow(probability_map[:, :, 250, 2], interpolation='none', cmap='gray')
                # plt.imshow(weight_map[:, :, 250, 2], interpolation='none', cmap='gray')
                pass
        else:
            probability_map = np.concatenate(predictions)

        end_time = time()

        with tf.device('/cpu:0'):
            elapsed_time = end_time - start_time
            logger.debug('  Elapsed Time (Seconds): %s', elapsed_time)
            self._write_inference_data(probability_map, filename, apply_path, version, self.options['rank'])

    def _run_train(self, logs_path, folder_name, training_dataset, validation_dataset,
                   summary_steps_per_epoch, epochs, l_r=0.001, optimizer='Adam', **kwargs):
        '''!
        Sets up and runs training session
        @param  logs_path               : str; path to logs
        @param  folder_name             : str
        @param  feed_dict_train         : dict
        @param  feed_dict_test          : dict
        @param  training_iterator       : tf.iterator
        @param  validation_iterator     : tf.iterator
        @param  summary_step            : int
        @param  l_r                     : float; Learning Rate for the optimizer
        @param  optimizer               : {'Adam', 'Momentum', 'Adadelta'}; Optimizer.
        @param epochs                   : int; number of epochs (for progress display)
        '''

        #TODO: generate graphs

        #start profiler
        profiler.start(logdir=os.path.abspath(os.path.join(logs_path, folder_name, 'profiler')))

        #save options for summary
        self.options['epochs'] = epochs
        self.options['learning_rate'] = l_r

        if not cfg.samples_per_volume * cfg.num_files % cfg.batch_size_train == 0:
            raise ValueError(
                f'cfg.samples_per_volume ({cfg.samples_per_volume}) * '+
                f'cfg.num_files ({cfg.num_files}) should be'+
                f' divisible by cfg.batch_size_train ({cfg.batch_size_train}). '
                f'Consider choosing samples per volume as multiple of batch size.'
            )
        iter_per_epoch = cfg.samples_per_volume * cfg.num_files // cfg.batch_size_train
        logger.debug('Iter per Epoch %s', iter_per_epoch)

        if self.options['do_finetune']:
            folder_name = folder_name + '-f'

        global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)

        train_path = os.path.abspath(os.path.join(logs_path, folder_name, 'model'))
        logger.debug('Training path: %s', train_path)
        if not os.path.exists(train_path):
            os.makedirs(train_path)

        # save the keras model
        self.model.save(os.path.join(train_path, 'keras_model.h5'))

        if not hasattr(self, 'optimizer'):
            self.optimizer = self._get_optimizer(optimizer, l_r, global_step)

        # save checkpoints
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=train_path, max_to_keep=3, checkpoint_name='weights')

        train_writer = tf.summary.create_file_writer(os.path.join(logs_path, folder_name, 'train'))
        valid_writer = tf.summary.create_file_writer(os.path.join(logs_path, folder_name, 'valid'))

        epoch_objective_avg = tf.keras.metrics.Mean()
        epoch_accuracy_avg = tf.keras.metrics.Mean()

        self.outputs['accuracy'] = metric.dice_coefficient_tf

        #make progress bar for epochs
        progress_bar_epochs = tqdm(
                    total=epochs,
                    desc=f'{folder_name} (train)',
                    position=1,
                    unit='epoch',
                    smoothing=0.9
                )

        for x_t, y_t in training_dataset:
            
            epoch = global_step.numpy() // iter_per_epoch

            #create progress bar and reset every epoch
            if global_step.numpy() % iter_per_epoch == 0:
                progress_bar_batches = tqdm(
                    total=iter_per_epoch,
                    desc=f'Epoch {epoch + 1}',
                    position=0,
                    smoothing=0.9
                )

            if global_step == 0:
                prediction, objective_train = self.training_step(x_t, y_t)
            else:
                # call profiler
                with profiler.Trace('train', step_num=global_step, _r=1):
                    # to the training step
                    prediction, objective_train = self.training_step(x_t, y_t)

            # update metrics
            if self.options['rank'] == 2:
                accuracy_train = self.outputs['accuracy'](y_t[:, :, :, 1], prediction[:, :, :, 1])
            else:
                accuracy_train = self.outputs['accuracy'](y_t[:, :, :, :, 1], prediction[:, :, :, :, 1])

            epoch_objective_avg.update_state(objective_train)
            epoch_accuracy_avg.update_state(accuracy_train)

            #call function when epoch ends (and at the start)
            if global_step.numpy() % iter_per_epoch == 0:
                self._end_of_epoch(checkpoint_manager, epoch, iter_per_epoch, validation_dataset, valid_writer)

                if self.options['do_finetune']:
                    for layer in self.model.layers[:min(epoch * 3, len(self.model.layers) // 2 - 3)]:
                        layer.trainable = False

                # use the epoch as step number, because it is more consistent
                self._summaries(x_t, y_t, prediction, epoch_objective_avg.result().numpy(),
                                epoch_accuracy_avg.result().numpy(),
                                epoch, train_writer, mode='train')

                if global_step.numpy() != 0:
                    epoch_objective_avg.reset_states()
                    epoch_accuracy_avg.reset_states()

                #update progress bars
                progress_bar_batches.set_postfix_str(
                    f'Obj.: {epoch_objective_avg.result().numpy():.4f}' +
                    f' Acc.: {epoch_accuracy_avg.result().numpy():.4f}'
                )
                if global_step.numpy() != 0:
                    progress_bar_epochs.update()

            #update progress bar
            progress_bar_batches.set_postfix_str(
                f'Obj.: {epoch_objective_avg.result().numpy():.4f}' +
                f' Acc.: {epoch_accuracy_avg.result().numpy():.4f}'
            )

            progress_bar_batches.update()

            global_step = global_step + 1

        # set last epoch
        epoch = global_step.numpy() // iter_per_epoch

        #update progressbar after final epoch
        progress_bar_epochs.update()
        logger.info(f'Done -- Finished training after {epoch} epochs /' +
            f'{global_step.numpy()} steps. Average Objective: {epoch_objective_avg.result().numpy()} (Train)')
        self._end_of_epoch(checkpoint_manager, epoch, iter_per_epoch, validation_dataset, valid_writer)
        # do final summary
        self._summaries(x_t, y_t, prediction, epoch_objective_avg.result().numpy(), epoch_accuracy_avg.result().numpy(),
                                epoch, train_writer, mode='train')

        # stop the profiler
        profiler.stop()

    def training_step(self, x_t, y_t):
        # enabling the computation of gradients
        with tf.GradientTape() as tape:
            prediction = self.model(x_t)
            if self.options['regularize'][0]:
                self.outputs['reg'] = tf.add_n(self.model.losses)
                objective_train = self.outputs['loss'](y_t, prediction) + self.outputs['reg']
            else:
                objective_train = self.outputs['loss'](y_t, prediction)

        gradients = tape.gradient(objective_train, self.model.trainable_variables)
        if cfg.do_gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, cfg.clipping_value)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return prediction, objective_train

    def _end_of_epoch(self, checkpoint_manager, step, iter_per_epoch, validation_dataset, valid_writer):
        logger.info(' Epoch: %s', step)
        checkpoint_manager.save(checkpoint_number=step)
        epoch_objective_avg = tf.keras.metrics.Mean()
        epoch_accuracy_avg = tf.keras.metrics.Mean()

        for x_v, y_v in validation_dataset:
            prediction = self.model(x_v, training=False)
            if self.options['regularize'][0]:
                self.outputs['reg'] = tf.add_n(self.model.losses)
                objective_validation = self.outputs['loss'](y_v, prediction) + self.outputs['reg']
            else:
                objective_validation = self.outputs['loss'](y_v, prediction)

            if self.options['rank'] == 2:
                accuracy_validation = self.outputs['accuracy'](y_v[:, :, :, 1], prediction[:, :, :, 1])
            else:
                accuracy_validation = self.outputs['accuracy'](y_v[:, :, :, :, 1], prediction[:, :, :, :, 1])

            epoch_objective_avg.update_state(objective_validation)
            epoch_accuracy_avg.update_state(accuracy_validation)

        logger.info(f' Epoch: {step} Average Objective: ' +
              f'{epoch_objective_avg.result().numpy():.4f} Average Accuracy: {epoch_accuracy_avg.result().numpy():.4f} (Valid)')
        self._summaries(x_v, y_v, prediction, epoch_objective_avg.result().numpy(),
                        epoch_accuracy_avg.result().numpy(), step, valid_writer, mode='valid')

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
            'optimizer' : self.optimizer._name,
            'drop_out_rate' : self.options['drop_out'][1],
            'regularize' : self.options['regularize'][0],
            'regularizer' : self.options['regularize'][1],
            'regularizer_param' : self.options['regularize'][2],
            'kernel_dim' : self.options['kernel_dims'][0]
        }
        for o in ['epochs', 'learning_rate', 'use_bias', 'loss', 'name', 'do_gradient_clipping', 'batch_normalization', 'activation', 'use_cross_hair']:
            hyperparameters[o] = self.options[o]
        # check the types
        for key, value in hyperparameters.items():
            if type(value) in [list]:
                hyperparameters[key] = str(value)
        return hyperparameters

    def _summaries(self, x, y, probabilities, objective, acccuracy, step, writer, max_image_output=2, histo_buckets=50, mode=None):

        predictions = tf.argmax(probabilities, -1)

        with writer.as_default():
            
            #save hyperparameters for tensorboard
            hp.hparams(self.get_hyperparameter_dict(), trial_id=cfg.trial_id)

            with tf.name_scope('01_Objective'):
                tf.summary.scalar('average_objective', objective, step=step)
                tf.summary.scalar('iteration_' + self.options['loss'], self.outputs['loss'](y, probabilities), step=step)
                if self.options['regularize'][0]:
                    tf.summary.scalar('iteration_' + self.options['regularize'][1], self.outputs['reg'], step=step)

            with tf.name_scope('02_Accuracy'):
                tf.summary.scalar('average_acccuracy', acccuracy, step=step)
                if mode != None:
                    tf.summary.scalar(f'average_acccuracy_{mode}', acccuracy, step=step)

            with tf.name_scope('03_Data_Statistics'):
                tf.summary.scalar('one_hot_max_train_img', tf.reduce_max(y), step=step)
                tf.summary.histogram('train_img', x, step, buckets=histo_buckets)
                tf.summary.histogram('label_img', y, step, buckets=histo_buckets)

            with tf.name_scope('04_Output_Statistics'):
                # tf.summary.histogram('logits', self.outputs['logits'], step, buckets=histo_buckets)
                tf.summary.histogram('probabilities', probabilities, step, buckets=histo_buckets)
                tf.summary.histogram('predictions', predictions, step, buckets=histo_buckets)

            with tf.name_scope('01_Input_and_Predictions'):
                if self.options['rank'] == 2:
                    if self.options['in_channels'] == 1:
                        tf.summary.image('train_img', tf.cast((tf.gather(x, [0, cfg.batch_size_train - 1]) + 1) * 255 / 2,
                                                              tf.uint8), step, max_image_output)
                    else:
                        for c in range(self.options['in_channels']):
                            tf.summary.image('train_img_c'+str(c), tf.expand_dims(tf.cast(
                                (tf.gather(x[:, :, :, c], [0, cfg.batch_size_train - 1]) + 1) * 255 / 2,
                                tf.uint8), axis=-1), step, max_image_output)

                    tf.summary.image('train_seg_lbl', tf.expand_dims(tf.cast(tf.argmax(
                        tf.gather(y, [0, cfg.batch_size_train - 1])
                        , -1) * (255 // (cfg.num_classes_seg - 1)), tf.uint8), axis=-1), step, max_image_output)
                    tf.summary.image('train_seg_pred', tf.expand_dims(tf.cast(
                        tf.gather(predictions, [0, cfg.batch_size_train - 1])
                        * (255 // (cfg.num_classes_seg - 1)), tf.uint8), axis=-1), step, max_image_output)

                else:
                    if self.options['in_channels'] == 1:
                        tf.summary.image('train_img', tf.cast(
                            (tf.gather(x[:, self.inputs['x'].shape[1] // 2, :, :], [0, cfg.batch_size_train - 1]) + 1) * 255 / 2,
                            tf.uint8), step, max_image_output)
                    else:
                        pass

                    tf.summary.image('train_seg_lbl', tf.expand_dims(tf.cast(tf.argmax(
                        tf.gather(y[:, self.inputs['x'].shape[1] // 2], [0, cfg.batch_size_train - 1]),
                        -1) * (255 // (cfg.num_classes_seg - 1)), tf.uint8), axis=-1), step, max_image_output)

                    tf.summary.image('train_seg_pred', tf.expand_dims(tf.cast(
                            tf.gather(predictions[:, self.inputs['x'].shape[1] // 2], [0, cfg.batch_size_train - 1])
                            * (255 // (cfg.num_classes_seg - 1)), tf.uint8), axis=-1), step, max_image_output)

            with tf.name_scope('02_Probabilities'):
                if self.options['rank'] == 2:
                    for c in range(cfg.num_classes_seg):
                        tf.summary.image('train_seg_prob_' + str(c), tf.expand_dims(tf.cast(
                            tf.gather(probabilities[:, :, :, c], [0, cfg.batch_size_train - 1])
                            * 255, tf.uint8), axis=-1), step, max_image_output)
                else:
                    for c in range(cfg.num_classes_seg):
                        tf.summary.image('train_seg_prob_class' + str(c), tf.expand_dims(tf.cast(
                            tf.gather(probabilities[:, self.inputs['x'].shape[1] // 2, :, :, c],
                                      [0, cfg.batch_size_train - 1])
                            * 255, tf.uint8), axis=-1), step, max_image_output)

            with tf.name_scope('03_Class_Labels'):
                if self.options['rank'] == 2:
                    pass
                else:
                    for c in range(cfg.num_classes_seg):
                        tf.summary.image('train_seg_lbl' + str(c), tf.expand_dims(tf.cast(
                            tf.gather(y[:, self.inputs['x'].shape[1] // 2, :, :, c], [0, cfg.batch_size_train - 1])
                            * 255, tf.uint8), axis=-1), step, max_image_output)

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

    @staticmethod
    def _write_inference_data(predictions, file_name, out_path, version, rank):
        import SimpleITK as sitk
        if not cfg.write_probabilities:
            predictions = np.argmax(predictions, -1)

        # Use a SimpleITK reader to load the nii images and labels for training
        folder, file_number = os.path.split(file_name)
        data_img = sitk.ReadImage(os.path.join(folder, (cfg.sample_file_name_prefix + file_number + cfg.file_suffix)))
        data_info = image.get_data_info(data_img)
        if cfg.adapt_resolution:
            target_info = {}
            target_info['target_spacing'] = cfg.target_spacing
            target_info['target_direction'] = cfg.target_direction
            target_info['target_size'] = cfg.target_size
            target_info['target_type_image'] = cfg.target_type_image
            target_info['target_type_label'] = cfg.target_type_image
            resampled_img = image.resample_sitk_image(data_img, target_info,
                                                      data_background_value=cfg.data_background_value,
                                                      do_adapt_resolution=cfg.adapt_resolution,
                                                      label_background_value=cfg.label_background_value,
                                                      do_augment=False)

            if rank == 3:
                z_original = resampled_img.GetSize()[-1]
                padded_extend = z_original + cfg.test_label_shape[0]
                least_number_of_samples = np.int(np.ceil(z_original / cfg.test_label_shape[0]))
                number_of_samples = 2 * least_number_of_samples - 1
                stiched_extend = np.ceil(number_of_samples/2) * cfg.test_label_shape[0]
                center = np.int(z_original//2 + cfg.test_label_shape[0] // 2 - (padded_extend - stiched_extend) // 2)
                if (z_original % 2) == 0:
                    predictions = predictions[center - (z_original // 2) + 1:]
                else:
                    predictions = predictions[center - (z_original // 2):]

            data_info['res_spacing'] = cfg.target_spacing
            data_info['res_origin'] = resampled_img.GetOrigin()
            data_info['res_direction'] = cfg.target_direction

        pred_img = image.np_array_to_sitk_image(predictions, data_info, cfg.label_background_value,
                                               cfg.adapt_resolution, cfg.target_type_label)

        name = Path(file_name).name
        pred_path = Path(out_path) / f'prediction-{name}-{version}{cfg.file_suffix}'
        sitk.WriteImage(pred_img, str(pred_path.absolute()))
