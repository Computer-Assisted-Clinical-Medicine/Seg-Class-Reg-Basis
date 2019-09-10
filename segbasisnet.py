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
        # tf.saved_model.save(self.model, train_path)
        # self.model.save(train_path, save_format="tf")
        tf.keras.experimental.export_saved_model(self.model, train_path)

        if not hasattr(self, 'optimizer'):
            self.optimizer = self._get_optimizer(optimizer, l_r, global_step)

        # save checkpoints in the variables folder of the keras model
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=os.path.join(train_path, "variables"), max_to_keep=3, checkpoint_name=folder_name)

        train_writer = tf.summary.create_file_writer(os.path.join(logs_path, folder_name, 'train'))
        with train_writer.as_default():
            tf.summary.trace_export(name="architecture", step=global_step, profiler_outdir=os.path.join(logs_path, folder_name, 'train'))
        valid_writer = tf.summary.create_file_writer(os.path.join(logs_path, folder_name, 'valid'))

        epoch_objective_avg = tf.keras.metrics.Mean()
        epoch_accuracy_avg = tf.keras.metrics.Mean()

        self.outputs['accuracy'] = Metric.dice_coefficient_tf

        for x_t, y_t in training_dataset:

            with tf.GradientTape() as tape:
                prediction = self.model(x_t)
                if self.options['regularize'][0]:
                    self.outputs['reg'] = tf.add_n(self.model.losses)
                    objective_train = self.outputs['loss'](prediction, y_t) + self.outputs['reg']
                else:
                    objective_train = self.outputs['loss'](prediction, y_t)
                accuracy_train = self.outputs['accuracy'](prediction, y_t)
                epoch_objective_avg.update_state(objective_train)
                epoch_accuracy_avg.update_state(accuracy_train)
            gradients = tape.gradient(objective_train, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            if global_step.numpy() % iter_per_epoch == 0 and global_step.numpy() // iter_per_epoch > 0:
                self._end_of_epoch(checkpoint_manager, global_step, iter_per_epoch, validation_dataset, valid_writer)

            # if true compute and write summaries
            elif global_step.numpy() % summary_step == 0:
                print('   Step: ', global_step.numpy(), ' Objective: ', epoch_objective_avg.result().numpy(),
                      ' Accuracy: ', epoch_accuracy_avg.result().numpy(), '(Train)')
                self._summaries(x_t, y_t, prediction, epoch_objective_avg.result().numpy(),
                                epoch_accuracy_avg.result().numpy(), global_step, train_writer)
                epoch_objective_avg.reset_states()
                epoch_accuracy_avg.reset_states()

            global_step = global_step + 1

        print('Done -- Finished training after', global_step.numpy() // iter_per_epoch,
              ' epochs /', global_step.numpy(), 'steps.', ' Average Objective: ', epoch_objective_avg.result().numpy(),
              '(Train)')
        self._end_of_epoch(checkpoint_manager, global_step, iter_per_epoch, validation_dataset, valid_writer)

    def _end_of_epoch(self, checkpoint_manager, global_step, iter_per_epoch, validation_dataset, valid_writer):
        print(' Epoch: ', global_step.numpy() // iter_per_epoch)
        checkpoint_manager.save(checkpoint_number=global_step.numpy() // iter_per_epoch)
        epoch_objective_avg = tf.keras.metrics.Mean()
        epoch_accuracy_avg = tf.keras.metrics.Mean()

        for x_v, y_v in validation_dataset:
            prediction = self.model(x_v)
            if self.options['regularize'][0]:
                self.outputs['reg'] = tf.add_n(self.model.losses)
                objective_validation = self.outputs['loss'](prediction, y_v) + self.outputs['reg']
            else:
                objective_validation = self.outputs['loss'](prediction, y_v)
            accuracy_validation = self.outputs['accuracy'](prediction, y_v)
            epoch_objective_avg.update_state(objective_validation)
            epoch_accuracy_avg.update_state(accuracy_validation)

        print(' Epoch: ', global_step.numpy() // iter_per_epoch, ' Average Objective: ',
              epoch_objective_avg.result().numpy(), ' Average Accuracy: ', epoch_accuracy_avg.result().numpy(), '(Valid)')
        self._summaries(x_v, y_v, prediction, epoch_objective_avg.result().numpy(),
                        epoch_accuracy_avg.result().numpy(), global_step, valid_writer)

    def _summaries(self, x, y, probabilities, objective, acccuracy, global_step, writer, max_image_output=2, histo_buckets=50):

        predictions = tf.argmax(probabilities, -1)

        with writer.as_default():
            with tf.name_scope('01_Objective'):
                tf.summary.scalar('average_objective', objective, step=global_step)
                tf.summary.scalar('iteration_' + self.options['loss'], self.outputs['loss'](probabilities, y), step=global_step)
                if self.options['regularize'][0]:
                    tf.summary.scalar('iteration_' + self.options['regularize'][1], self.outputs['reg'], step=global_step)

            with tf.name_scope('02_Accuracy'):
                tf.summary.scalar('average_acccuracy', acccuracy, step=global_step)

            with tf.name_scope('03_Data_Statistics'):
                tf.summary.scalar('one_hot_max_train_img', tf.reduce_max(y), step=global_step)
                tf.summary.histogram('train_img', x, global_step, buckets=histo_buckets)
                tf.summary.histogram('label_img', y, global_step, buckets=histo_buckets)

            with tf.name_scope('03_Output_Statistics'):
                # tf.summary.histogram('logits', self.outputs['logits'], global_step, buckets=histo_buckets)
                tf.summary.histogram('probabilities', probabilities, global_step, buckets=histo_buckets)
                tf.summary.histogram('predictions', predictions, global_step, buckets=histo_buckets)

            with tf.name_scope('01_Input_and_Predictions'):
                if self.options['rank'] == 2:
                    if self.options['in_channels'] > 1:
                        if self.options['in_channels'] > 2:
                            tf.summary.image('train_img', tf.cast((tf.gather(x, [0, cfg.batch_size - 1])
                                                                   [:, :, :, ::self.options['in_channels'] // 2] + 1) * 255 / 2,
                                                                  tf.uint8), global_step, max_image_output)
                        else:
                            tf.summary.image('train_img_c1', tf.expand_dims(
                                tf.cast((tf.gather(x[:, :, :, 0], [0, cfg.batch_size - 1]) + 1) * 255 / 2,
                                        tf.uint8), axis=-1), global_step, max_image_output)
                            tf.summary.image('train_img_c2',
                                             tf.expand_dims(
                                                 tf.cast(
                                                     (tf.gather(x[:, :, :, 1],
                                                                [0, cfg.batch_size - 1]) + 1) * 255 / 2,
                                                     tf.uint8), axis=-1), global_step, max_image_output)
                    else:
                        tf.summary.image('train_img', tf.cast((tf.gather(x, [0, cfg.batch_size - 1]) + 1) * 255 / 2,
                            tf.uint8), global_step, max_image_output)

                    if self.options['out_channels'] == 2:
                        tf.summary.image('train_seg_lbl', tf.expand_dims(
                            tf.cast(tf.gather(y[:, :, :, 1], [0, cfg.batch_size - 1]) * 255, tf.uint8), axis=-1), global_step, max_image_output)
                        tf.summary.image('train_seg_prob', tf.expand_dims(
                            tf.cast(tf.gather(probabilities[:, :, :, 1], [0, cfg.batch_size - 1]) * 255, tf.uint8), axis=-1), global_step, max_image_output)
                        tf.summary.image('train_seg_pred', tf.expand_dims(
                            tf.cast(tf.gather(predictions, [0, cfg.batch_size - 1]) * 255, tf.uint8),
                            axis=-1), global_step, max_image_output)
                else:
                    if self.options['in_channels'] == 1:
                        tf.summary.image('train_img', tf.cast((tf.gather(x[:, self.inputs['x'].shape[1] // 2, :, :], [0, cfg.batch_size - 1]) + 1) * 255 / 2,
                                                              tf.uint8), global_step, max_image_output)
                    if self.options['out_channels'] == 2:
                        tf.summary.image('train_seg_lbl', tf.expand_dims(
                            tf.cast(tf.gather(tf.squeeze(y[:, self.inputs['x'].shape[1] // 2, :, :, 1]), [0, cfg.batch_size - 1]) * 255, tf.uint8), axis=-1),
                                         global_step, max_image_output)
                        tf.summary.image('train_seg_prob', tf.expand_dims(
                            tf.cast(tf.gather(probabilities[:, self.inputs['x'].shape[1] // 2, :, :, 1], [0, cfg.batch_size - 1]) * 255, tf.uint8),
                            axis=-1), global_step, max_image_output)
                        tf.summary.image('train_seg_pred', tf.expand_dims(
                            tf.cast(tf.gather(predictions[:, self.inputs['x'].shape[1] // 2, :], [0, cfg.batch_size - 1]) * 255, tf.uint8),
                            axis=-1), global_step, max_image_output)


                #
                # else:
                #     tf.summary.scalar('OneHotMax-Img', tf.reduce_max(self.inputs['y']),
                #                       collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                #     help_img = self.inputs['y'][:, :, :, 1]
                #     for i in range(2, self.options['out_channels']):
                #         help_img = tf.expand_dims(tf.add(help_img, tf.multiply(self.inputs['y'][:, :, :, i], i)), -1)
                #     tf.summary.scalar('IntMax-Img', tf.reduce_max(help_img),
                #                       collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                #     tf.summary.image('train_seg_lbl',
                #                      tf.cast(tf.gather(help_img, [0, cfg.batch_size - 1]) * (255 // (self.n_classes - 1)),
                #                              tf.uint8), 2,
                #                      collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                #     tf.summary.image('train_seg_pred', tf.expand_dims(
                #         tf.cast(
                #             tf.gather(self.outputs['probabilitiess'], [0, cfg.batch_size - 1]) * (255 // (self.n_classes - 1)),
                #             tf.uint8), axis=-1), 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                #     tf.summary.image('train_seg_log_object_0',
                #                      tf.expand_dims(tf.gather(self.outputs['logits'][:, :, :, 1], [0, cfg.batch_size - 1]),
                #                                     axis=-1),
                #                      2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                #     tf.summary.image('train_seg_log_object_1',
                #                      tf.expand_dims(tf.gather(self.outputs['logits'][:, :, :, 2], [0, cfg.batch_size - 1]),
                #                                     axis=-1),
                #                      2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                #     tf.summary.image('train_seg_prob_0', tf.expand_dims(
                #         tf.cast(tf.gather(self.outputs['probabilities'][:, :, :, 1], [0, cfg.batch_size - 1]) * 255,
                #                 tf.uint8), axis=-1),
                #                      2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                #     tf.summary.image('train_seg_prob_1', tf.expand_dims(
                #         tf.cast(tf.gather(self.outputs['probabilities'][:, :, :, 2], [0, cfg.batch_size - 1]) * 255,
                #                 tf.uint8), axis=-1),
                #                      2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                #
                #     tf.summary.image('train_seg_log_background', tf.expand_dims(tf.gather(self.outputs['logits'][:, :, :, 0],
                #                                                                       [0, cfg.batch_size - 1]), axis=-1), 2,
                #                  collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])

            with tf.name_scope('02_Logits_and_Probabilities'):
                pass

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

    def _write_inference_data(self, predictions, file_name, out_path, version):
        raise NotImplementedError('not implemented')

