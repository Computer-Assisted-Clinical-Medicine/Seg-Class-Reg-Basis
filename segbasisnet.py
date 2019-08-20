import tensorflow as tf
import numpy as np
from time import time
from . import config as cfg
from .NetworkBasis.network import Network
from .NetworkBasis import loss as Loss
from .NetworkBasis import metric as Metric
import os
import multiprocessing


class SegBasisNet(Network):

    def _set_up_inputs(self):
        if self.options['is_training']:
            self.inputs['x'] = tf.keras.Input(shape=cfg.train_input_shape, batch_size=cfg.batch_size, dtype=cfg.dtype)
        else:
            self.inputs['x'] = tf.keras.Input(shape=cfg.cfg.test_data_shape, batch_size=cfg.batch_size, dtype=cfg.dtype)

        self.options['out_channels'] = cfg.num_classes_seg

    def _get_loss(self):
        '''!
        Returns loss depending on `self.options['loss']``.

        self.options['loss'] should be in {'DICE', 'TVE', 'CEL', 'WCEL'}.
        @returns @b loss : function
        '''
        if self.options['loss'] == 'DICE':
            loss = Loss.dice_loss

        elif self.options['loss'] == 'TVE':
            loss = Loss.tversky_loss

        elif self.options['loss'] == 'GDL':
            loss = Loss.generalized_dice_loss

        elif self.options['loss'] == 'CEL':
            if self.options['out_channels'] > 2:
                loss = Loss.categorical_cross_entropy_loss
            else:
                loss = Loss.binary_cross_entropy_loss

        elif self.options['loss'] == 'WCEL':
            # ToDo: update this
            if self.options['out_channels'] > 2:
                loss = Loss.weighted_cross_entropy_loss_with_softmax(self.outputs['logits'], self.inputs['y'], self.inputs['x'],
                                                                     cfg.basis_factor, cfg.tissue_factor,
                                                                     cfg.contour_factor, cfg.tissue_threshold,
                                                                     cfg.max_weight)
            else:
                loss = Loss.weighted_cross_entropy_loss_with_sigmoid(self.outputs['logits'], self.inputs['y'], self.inputs['x'],
                                                                     cfg.basis_factor, cfg.tissue_factor,
                                                                     cfg.contour_factor, cfg.tissue_threshold,
                                                                     cfg.max_weight)

        else:
            raise ValueError(self.options['loss'], 'is not a supported loss function.')

        return loss

    def _run_apply(self, version, apply_path, application_dataset, filename):

        if not os.path.exists(apply_path):
            os.makedirs(apply_path)

        predictions = []
        start_time = time()

        for x in application_dataset:
            prediction = self.model(x)
            predictions.append(prediction)

        end_time = time()

        with tf.device('/cpu:0'):
            elapsed_time = end_time - start_time
            print('  Elapsed Time (Seconds): ', elapsed_time)

            predictions = np.concatenate(predictions)
            self._write_inference_data(predictions, filename, apply_path, version)

    def _run_train(self, logs_path, folder_name, training_dataset, validation_dataset, op_parallelism_threads=-1,
                   summary_step=10, write_step=1500, l_r=0.001, optimizer='Adam'):
        '''!
        Sets up and runs training session
        @param  logs_path               : str; path to logs
        @param  folder_name             : str
        @param  feed_dict_train         : dict
        @param  feed_dict_test          : dict
        @param  training_iterator       : tf.iterator
        @param  validation_iterator     : tf.iterator
        @param  summary_step            : int
        @param  write_step              : int
        @param  l_r                     : float; Learning Rate for the optimizer
        @param  optimizer               : {'Adam', 'Momentum', 'Adadelta'}; Optimizer.
        '''

        iter_per_epoch = cfg.samples_per_volume * cfg.num_files // cfg.batch_size
        print('Iter per Epoch', iter_per_epoch)

        if self.options['do_finetune']:
            folder_name = folder_name + '-f'

        global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)

        train_path = os.path.join(logs_path, folder_name, 'model')
        print(train_path)

        # save the keras model
        tf.keras.experimental.export_saved_model(self.model, train_path)

        with tf.name_scope('objective'):
            self.outputs['objective'] = self._get_loss()

        if not hasattr(self, 'optimizer'):
            self.optimizer = self._get_optimizer(optimizer, l_r, global_step)

        # save checkpoints in the variables folder of the keras model
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=os.path.join(train_path, "variables"), max_to_keep=3, checkpoint_name=folder_name)

        train_writer = tf.summary.create_file_writer(os.path.join(logs_path, folder_name, 'train'))
        valid_writer = tf.summary.create_file_writer(os.path.join(logs_path, folder_name, 'valid'))

        epoch_loss_avg = tf.keras.metrics.Mean()

        for x_t, y_t in training_dataset:

            with tf.GradientTape() as tape:
                prediction = self.model(x_t)
                objective_train = self.outputs['objective'](prediction, y_t)
                epoch_loss_avg.update_state(objective_train)
            gradients = tape.gradient(objective_train, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            if global_step.numpy() % iter_per_epoch == 0 and global_step.numpy() // iter_per_epoch > 0:
                self._end_of_epoch(checkpoint_manager, global_step, iter_per_epoch, epoch_loss_avg, validation_dataset)

            # if true compute and write summaries
            elif global_step.numpy() % summary_step == 0:
                print('   Step: ', global_step.numpy(), ' Objective: ', objective_train.numpy(), '(Train)')
                with train_writer.as_default():
                    tf.summary.scalar('loss', objective_train, step=global_step)

            global_step = global_step + 1

        self._end_of_epoch(checkpoint_manager, global_step, iter_per_epoch, epoch_loss_avg, validation_dataset, valid_writer)
        print('Done -- Finished training after', global_step.numpy() // iter_per_epoch,
              ' epochs /', global_step.numpy(), 'steps.', ' Average Objective: ', epoch_loss_avg.result().numpy(), '(Train)')

    def _end_of_epoch(self, checkpoint_manager, global_step, iter_per_epoch, epoch_loss_avg, validation_dataset, valid_writer):
        print(' Epoch: ', global_step.numpy() // iter_per_epoch)
        checkpoint_manager.save(checkpoint_number=global_step.numpy() // iter_per_epoch)
        epoch_loss_avg.reset_states()

        for x_v, y_v in validation_dataset:
            prediction = self.model(x_v)
            objective_validation = self.outputs['objective'](prediction, y_v)
            epoch_loss_avg.update_state(objective_validation)

        print(' Epoch: ', global_step.numpy() // iter_per_epoch, ' Average Objective: ',
              epoch_loss_avg.result().numpy(), '(Valid)')
        with valid_writer.as_default():
            tf.summary.scalar('loss', epoch_loss_avg.result(), step=global_step)

    def _select_final_activation(self):
        # http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/
        if self.options['out_channels'] > 2 or self.options['loss'] in ['DICE', 'TVE', 'GDL']:
            # Dice and Tversky require SoftMax
            return 'softmax'
        elif self.options['out_channels'] == 2 and self.options['loss'] in ['CEL', 'WCEL']:
            return 'sigmoid'
        else:
            raise ValueError(self.options['loss'],
                             'is not a supported loss function or cannot combined with ',
                             self.options['out_channels'], 'output channels.')

    def _read_data_for_inference(self, file, mode):
        raise NotImplementedError('not implemented')

    def _write_inference_data(self, predictions, data_info, file_number, out_path, version):
        raise NotImplementedError('not implemented')

