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

    def _load_collections(self):
        self.outputs['probabilities'] = tf.get_collection("probabilities")[0]
        self.outputs['predictions'] = tf.get_collection("predictions")[0]

    def _add_data_summaries(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope('data'):
                tf.summary.histogram('input', self.inputs['x'], collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                help_img = self.inputs['y'][:, :, :, 1]
                for i in range(2, self.options['out_channels']):
                    help_img = tf.expand_dims(tf.add(help_img, tf.multiply(self.inputs['y'][:, :, :, i], i)), -1)
                tf.summary.histogram('label', help_img,
                                     collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.histogram('predictions', self.outputs['predictions'], collections=[tf.GraphKeys.SUMMARIES,
                                                                                              'vald_summaries'])

    def _set_up_prediction(self):
        '''!
        Sets up predictions
        '''
        # Prediction
        with tf.variable_scope('prediction'):
            if self.options['out_channels'] == 1:
                self.outputs['predictions'] = tf.cast(tf.greater_equal(self.outputs['logits'], 0.5), tf.int32)
            else:
                # http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/
                if self.options['out_channels'] > 2 or self.options['loss'] == 'DICE' or self.options['loss'] == 'TVE':
                    # Dice and Tversky require SoftMax
                    self.outputs['probabilities'] = tf.nn.softmax(self.outputs['logits'])
                else:
                    self.outputs['probabilities'] = tf.nn.sigmoid(self.outputs['logits'])

                self.outputs['predictions'] = tf.cast(tf.argmax(self.outputs['probabilities'], axis=-1), tf.int32)

        # add to collections to make retrievable from graph
        tf.add_to_collection("probabilities", self.outputs['probabilities'])
        tf.add_to_collection("predictions", self.outputs['predictions'])

    def _get_loss(self):
        '''!
                Returns loss depending on `self.options['loss']``.

                self.options['loss'] should be in {'DICE', 'TVE', 'CEL', 'WCEL'}.
                @returns @b loss : tf.float
                '''
        # Assert that dimensions match!
        # assert self.outputs['probabilities'].shape.as_list()[1:-1] == self.inputs['y'].shape.as_list()[1:-1]

        # Loss
        if self.options['loss'] == 'DICE':
            loss = Loss.dice_loss

        elif self.options['loss'] == 'TVE':
            loss = Loss.tversky_loss(self.outputs['probabilities'], self.inputs['y'], cfg.tversky_alpha, cfg.tversky_beta)

        elif self.options['loss'] == 'GDL':
            loss = Loss.generalized_dice_loss(self.outputs['probabilities'], self.inputs['y'])

        elif self.options['loss'] == 'CEL':
            if self.options['out_channels'] > 2:
                loss = Loss.cross_entropy_loss_with_softmax(logits=self.outputs['logits'], labels=self.inputs['y'])
            else:
                loss = Loss.cross_entropy_loss_with_sigmoid(logits=self.outputs['logits'], labels=self.inputs['y'])

        elif self.options['loss'] == 'WCEL':
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

    def _set_up_training_image_summaries(self):
        with tf.device('/cpu:0'):
            if self.options['in_channels'] > 1:
                if self.options['in_channels'] > 2:
                    tf.summary.image('train_img', tf.cast((tf.gather(self.inputs['x'], [0, cfg.batch_size-1])
                                    [:, :, :, ::self.options['in_channels']//2]+1) * 255/2, tf.uint8),
                                     2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                else:
                    tf.summary.image('train_img_c1', tf.expand_dims(
                        tf.cast((tf.gather(self.inputs['x'][:, :, :, 0], [0, cfg.batch_size - 1]) + 1) * 255 / 2, tf.uint8),
                        axis=-1), 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                    tf.summary.image('train_img_c2',
                                     tf.expand_dims(
                                         tf.cast(
                                             (tf.gather(self.inputs['x'][:, :, :, 1], [0, cfg.batch_size - 1]) + 1) * 255 / 2,
                                             tf.uint8),
                                         axis=-1), 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
            else:
                tf.summary.image('train_img', tf.cast(
                    (tf.gather(self.inputs['x'], [0, cfg.batch_size - 1]) + 1) * 255 / 2,
                    tf.uint8), 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])

            if self.options['out_channels'] == 2:
                tf.summary.image('train_seg_lbl', tf.expand_dims(
                    tf.cast(tf.gather(self.inputs['y'][:, :, :, 1], [0, cfg.batch_size - 1]) * 255, tf.uint8), axis=-1),
                                 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.image('train_seg_pred', tf.expand_dims(
                    tf.cast(tf.gather(self.outputs['predictions'], [0, cfg.batch_size - 1]) * 255, tf.uint8), axis=-1),
                                 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.image('train_seg_log_object',
                                 tf.expand_dims(tf.gather(self.outputs['logits'][:, :, :, 1], [0, cfg.batch_size - 1]),
                                                axis=-1),
                                 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.image('train_seg_prob', tf.expand_dims(
                    tf.cast(tf.gather(self.outputs['probabilities'][:, :, :, 1], [0, cfg.batch_size - 1]) * 255,
                            tf.uint8), axis=-1),
                                 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
            else:
                tf.summary.scalar('OneHotMax-Img', tf.reduce_max(self.inputs['y']), collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                help_img = self.inputs['y'][:, :, :, 1]
                for i in range(2, self.options['out_channels']):
                    help_img = tf.expand_dims(tf.add(help_img, tf.multiply(self.inputs['y'][:, :, :, i], i)), -1)
                tf.summary.scalar('IntMax-Img', tf.reduce_max(help_img),
                                  collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.image('train_seg_lbl', tf.cast(tf.gather(help_img, [0, cfg.batch_size-1]) * (255 // (self.n_classes-1)), tf.uint8), 2,
                                 collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.image('train_seg_pred', tf.expand_dims(
                    tf.cast(tf.gather(self.outputs['predictions'], [0, cfg.batch_size - 1]) * (255 // (self.n_classes-1)),
                            tf.uint8), axis=-1), 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.image('train_seg_log_object_0',
                                 tf.expand_dims(tf.gather(self.outputs['logits'][:, :, :, 1], [0, cfg.batch_size - 1]),
                                                axis=-1),
                                 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.image('train_seg_log_object_1',
                                 tf.expand_dims(tf.gather(self.outputs['logits'][:, :, :, 2], [0, cfg.batch_size - 1]),
                                                axis=-1),
                                 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.image('train_seg_prob_0', tf.expand_dims(
                    tf.cast(tf.gather(self.outputs['probabilities'][:, :, :, 1], [0, cfg.batch_size - 1]) * 255,
                            tf.uint8), axis=-1),
                                 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.image('train_seg_prob_1', tf.expand_dims(
                    tf.cast(tf.gather(self.outputs['probabilities'][:, :, :, 2], [0, cfg.batch_size - 1]) * 255,
                            tf.uint8), axis=-1),
                                 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])

            tf.summary.image('train_seg_log_background', tf.expand_dims(tf.gather(self.outputs['logits'][:, :, :, 0],
                        [0, cfg.batch_size - 1]), axis=-1), 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])

    def _set_up_testing_image_summaries(self):
        with tf.device('/cpu:0'):
            if self.options['in_channels'] > 1:
                if self.options['in_channels'] > 2:
                    tf.summary.image('test_img', tf.cast((tf.gather(self.inputs['x'], [0, cfg.batch_size - 1])
                                                           [:, :, :, ::self.options['in_channels'] // 2] + 1) * 255 / 2,
                                                          tf.uint8),
                                     2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                else:
                    tf.summary.image('test_img_c1', tf.expand_dims(tf.cast((self.inputs['x'][:, :, :, 0] + 1) * 255 / 2,
                                tf.uint8),axis=-1), 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                    tf.summary.image('test_img_c2',
                                     tf.expand_dims(tf.cast((self.inputs['x'][:, :, :, 1] + 1) * 255 / 2,
                                             tf.uint8), axis=-1), 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
            else:
                tf.summary.image('test_img', tf.cast(
                    (self.inputs['x'] + 1) * 255 / 2,
                    tf.uint8), 1, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])

            if self.options['out_channels'] == 2:
                tf.summary.image('test_seg_pred', tf.expand_dims(tf.cast(self.outputs['predictions'] * 255, tf.uint8), axis=-1),
                             1)
                tf.summary.image('test_seg_lbl', tf.expand_dims(tf.cast(self.inputs['y'][:, :, :, 1] * 255, tf.uint8), axis=-1), 1)
                tf.summary.image('test_seg_prob',
                             tf.expand_dims(tf.cast(self.outputs['probabilities'][:, :, :, 1] * 255, tf.uint8), axis=-1), 1)
                tf.summary.image('test_seg_log_object',
                             tf.expand_dims(self.outputs['logits'][:, :, :, 1], axis=-1), 1)
            else:
                tf.summary.scalar('OneHotMax-Img', tf.reduce_max(self.inputs['y']),
                                  collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                help_img = self.inputs['y'][:, :, :, 1]
                for i in range(2, self.options['out_channels']):
                    help_img = tf.expand_dims(tf.add(help_img, tf.multiply(self.inputs['y'][:, :, :, i], i)), -1)
                tf.summary.scalar('IntMax-Img', tf.reduce_max(help_img),
                                  collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.image('test_seg_lbl',
                                 tf.cast(help_img * (255 // (self.n_classes - 1)),
                                         tf.uint8), 1, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.image('test_seg_pred', tf.expand_dims(
                    tf.cast(self.outputs['predictions'] * (255 // (self.n_classes - 1)),
                        tf.uint8), axis=-1), 1, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.image('test_seg_log_object_0',
                                 tf.expand_dims(self.outputs['logits'][:, :, :, 1],
                                                axis=-1),
                                 1, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.image('test_seg_log_object_1',
                                 tf.expand_dims(self.outputs['logits'][:, :, :, 2],
                                                axis=-1),
                                 1, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.image('test_seg_prob_0', tf.expand_dims(
                    tf.cast(self.outputs['probabilities'][:, :, :, 1] * 255,
                            tf.uint8), axis=-1), 1, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                tf.summary.image('test_seg_prob_1', tf.expand_dims(
                    tf.cast(self.outputs['probabilities'][:, :, :, 2] * 255,
                            tf.uint8), axis=-1), 1, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])

            tf.summary.image('test_seg_log_background',
                             tf.expand_dims(self.outputs['logits'][:, :, :, 0], axis=-1), 1)

    def _set_up_error(self):
        self.outputs['error'] = []
        for i in range(1, self.options['out_channels']):
            self.outputs['error'].append(1 - Metric.dice_coefficient_tf(tf.expand_dims(
                self.outputs['probabilities'][:, :, :, i], -1), tf.expand_dims(self.inputs['y'][:, :, :, i], -1)))
            with tf.device('/cpu:0'):
                tf.summary.scalar('dice_'+str(i), self.outputs['error'][i-1], collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])

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

            global_step = global_step + 1

        self._end_of_epoch(checkpoint_manager, global_step, iter_per_epoch, epoch_loss_avg, validation_dataset)
        print('Done -- Finished training after', global_step.numpy() // iter_per_epoch,
              ' epochs /', global_step.numpy(), 'steps.', ' Average Objective: ', epoch_loss_avg.result().numpy(), '(Train)')

    def _end_of_epoch(self, checkpoint_manager, global_step, iter_per_epoch, epoch_loss_avg, validation_dataset):
        print(' Epoch: ', global_step.numpy() // iter_per_epoch)
        checkpoint_manager.save(checkpoint_number=global_step.numpy() // iter_per_epoch)
        epoch_loss_avg.reset_states()

        for x_v, y_v in validation_dataset:
            prediction = self.model(x_v)
            objective_validation = self.outputs['objective'](prediction, y_v)
            epoch_loss_avg.update_state(objective_validation)

        print(' Epoch: ', global_step.numpy() // iter_per_epoch, ' Average Objective: ',
              epoch_loss_avg.result().numpy(), '(Valid)')

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

    @staticmethod
    def _get_optimizer(optimizer, l_r, global_step):
        if optimizer == 'Adam':
            return tf.optimizers.Adam(learning_rate=l_r, epsilon=1e-3)
        elif optimizer == 'Momentum':
            mom = 0.9
            learning_rate = tf.train.exponential_decay(l_r, global_step, 6000, 0.96, staircase=True)
            return tf.train.MomentumOptimizer(learning_rate, momentum=mom)
        elif optimizer == 'Adadelta':
            return tf.optimizers.AdamAdadelta(learning_rate=l_r)

    def _read_data_for_inference(self, file, mode):
        raise NotImplementedError('not implemented')

    def _write_inference_data(self, predictions, data_info, file_number, out_path, version):
        raise NotImplementedError('not implemented')

