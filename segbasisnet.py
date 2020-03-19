import tensorflow as tf
import numpy as np
from time import time
from . import config as cfg
from .NetworkBasis import image
from .NetworkBasis.network import Network
from .NetworkBasis import loss
from .NetworkBasis import metric
import os


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
            return loss.equalized_catergorical_cross_entropy

        elif self.options['loss'] == 'NCEL':
            return loss.normalized_catergorical_cross_entropy

        elif self.options['loss'] == 'WCEL':
            return loss.weighted_catergorical_cross_entropy

        elif self.options['loss'] == 'ECEL-FNR':
            return loss.equalized_catergorical_cross_entropy_with_fnr

        elif self.options['loss'] == 'WCEL-FPR':
                return loss.weighted_categorical_crossentropy_with_fpr_loss

        elif self.options['loss'] == 'GCEL':
                return loss.generalized_catergorical_cross_entropy

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
            print('  Elapsed Time (Seconds): ', elapsed_time)
            self._write_inference_data(probability_map, filename, apply_path, version, self.options['rank'])

    def _run_train(self, logs_path, folder_name, training_dataset, validation_dataset,
                   summary_steps_per_epoch, l_r=0.001, optimizer='Adam'):
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
        '''

        iter_per_epoch = cfg.samples_per_volume * cfg.num_files // cfg.batch_size_train
        print('Iter per Epoch', iter_per_epoch)
        summary_step = iter_per_epoch // summary_steps_per_epoch

        if self.options['do_finetune']:
            folder_name = folder_name + '-f'

        global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)

        train_path = os.path.abspath(os.path.join(logs_path, folder_name, 'model'))
        print(train_path)
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
        # with train_writer.as_default():
        #     tf.summary.trace_export(name="architecture", step=global_step, profiler_outdir=os.path.join(logs_path, folder_name, 'train'))
        valid_writer = tf.summary.create_file_writer(os.path.join(logs_path, folder_name, 'valid'))

        epoch_objective_avg = tf.keras.metrics.Mean()
        epoch_accuracy_avg_art = tf.keras.metrics.Mean()
        epoch_accuracy_avg_vein = tf.keras.metrics.Mean()


        self.outputs['accuracy'] = metric.dice_coefficient_tf

        for x_t, y_t in training_dataset:

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

            if self.options['rank'] == 2:
                accuracy_train_art = self.outputs['accuracy'](y_t[:, :, :, 1], prediction[:, :, :, 1])
                accuracy_train_vein = self.outputs['accuracy'](y_t[:, :, :, 2], prediction[:, :, :, 2])
            else:
                accuracy_train_art = self.outputs['accuracy'](y_t[:, :, :, :, 1], prediction[:, :, :, :, 1])
                accuracy_train_vein = self.outputs['accuracy'](y_t[:, :, :, :, 2], prediction[:, :, :, :, 2])

            epoch_objective_avg.update_state(objective_train)
            epoch_accuracy_avg_art.update_state(accuracy_train_art)
            epoch_accuracy_avg_vein.update_state(accuracy_train_vein)

            if global_step.numpy() % iter_per_epoch == 0 and global_step.numpy() // iter_per_epoch > 0:
                self._end_of_epoch(checkpoint_manager, global_step, iter_per_epoch, validation_dataset, valid_writer)

                if self.options['do_finetune']:
                    for layer in self.model.layers[:min(global_step.numpy() // iter_per_epoch * 3, len(self.model.layers) // 2 - 3)]:
                        layer.trainable = False

            # if true compute and write summaries
            elif global_step.numpy() % summary_step == 0:
                print('   Step: ', global_step.numpy(), ' Objective: ', epoch_objective_avg.result().numpy(),
                      ' Accuracy (V/A): ', epoch_accuracy_avg_vein.result().numpy(), epoch_accuracy_avg_art.result().numpy(), '(Train)')
                self._summaries(x_t, y_t, prediction, epoch_objective_avg.result().numpy(),
                                epoch_accuracy_avg_art.result().numpy(), epoch_accuracy_avg_vein.result().numpy(),
                                global_step, train_writer)
                epoch_objective_avg.reset_states()
                epoch_accuracy_avg_art.reset_states()
                epoch_accuracy_avg_vein.reset_states()

            global_step = global_step + 1

        print('Done -- Finished training after', global_step.numpy() // iter_per_epoch,
              ' epochs /', global_step.numpy(), 'steps.', ' Average Objective: ', epoch_objective_avg.result().numpy(),
              '(Train)')
        self._end_of_epoch(checkpoint_manager, global_step, iter_per_epoch, validation_dataset, valid_writer)

    def _end_of_epoch(self, checkpoint_manager, global_step, iter_per_epoch, validation_dataset, valid_writer):
        print(' Epoch: ', global_step.numpy() // iter_per_epoch)
        checkpoint_manager.save(checkpoint_number=global_step.numpy() // iter_per_epoch)
        epoch_objective_avg = tf.keras.metrics.Mean()
        epoch_accuracy_avg_art = tf.keras.metrics.Mean()
        epoch_accuracy_avg_vein = tf.keras.metrics.Mean()

        for x_v, y_v in validation_dataset:
            prediction = self.model(x_v, training=False)
            if self.options['regularize'][0]:
                self.outputs['reg'] = tf.add_n(self.model.losses)
                objective_validation = self.outputs['loss'](y_v, prediction) + self.outputs['reg']
            else:
                objective_validation = self.outputs['loss'](y_v, prediction)

            if self.options['rank'] == 2:
                accuracy_validation_art = self.outputs['accuracy'](y_v[:, :, :, 1], prediction[:, :, :, 1])
                accuracy_validation_vein = self.outputs['accuracy'](y_v[:, :, :, 2], prediction[:, :, :, 2])
            else:
                accuracy_validation_art = self.outputs['accuracy'](y_v[:, :, :, :, 1], prediction[:, :, :, :, 1])
                accuracy_validation_vein = self.outputs['accuracy'](y_v[:, :, :, :, 2], prediction[:, :, :, :, 2])

            epoch_objective_avg.update_state(objective_validation)
            epoch_accuracy_avg_art.update_state(accuracy_validation_art)
            epoch_accuracy_avg_vein.update_state(accuracy_validation_vein)

        print(' Epoch: ', global_step.numpy() // iter_per_epoch, ' Average Objective: ',
              epoch_objective_avg.result().numpy(), ' Average Accuracy (V/A): ', epoch_accuracy_avg_vein.result().numpy(), epoch_accuracy_avg_art.result().numpy(), '(Valid)')
        self._summaries(x_v, y_v, prediction, epoch_objective_avg.result().numpy(),
                        epoch_accuracy_avg_art.result().numpy(), epoch_accuracy_avg_vein.result().numpy(), global_step, valid_writer)

    def _summaries(self, x, y, probabilities, objective, acccuracy_art, acccuracy_vein, global_step, writer, max_image_output=2, histo_buckets=50):

        predictions = tf.argmax(probabilities, -1)

        with writer.as_default():
            with tf.name_scope('01_Objective'):
                tf.summary.scalar('average_objective', objective, step=global_step)
                tf.summary.scalar('iteration_' + self.options['loss'], self.outputs['loss'](y, probabilities), step=global_step)
                if self.options['regularize'][0]:
                    tf.summary.scalar('iteration_' + self.options['regularize'][1], self.outputs['reg'], step=global_step)

            with tf.name_scope('02_Accuracy'):
                tf.summary.scalar('average_acccuracy_art', acccuracy_art, step=global_step)
                tf.summary.scalar('average_acccuracy_vein', acccuracy_vein, step=global_step)

            with tf.name_scope('03_Data_Statistics'):
                tf.summary.scalar('one_hot_max_train_img', tf.reduce_max(y), step=global_step)
                tf.summary.histogram('train_img', x, global_step, buckets=histo_buckets)
                tf.summary.histogram('label_img', y, global_step, buckets=histo_buckets)

            with tf.name_scope('04_Output_Statistics'):
                # tf.summary.histogram('logits', self.outputs['logits'], global_step, buckets=histo_buckets)
                tf.summary.histogram('probabilities', probabilities, global_step, buckets=histo_buckets)
                tf.summary.histogram('predictions', predictions, global_step, buckets=histo_buckets)

            with tf.name_scope('01_Input_and_Predictions'):
                if self.options['rank'] == 2:
                    if self.options['in_channels'] == 1:
                        tf.summary.image('train_img', tf.cast((tf.gather(x, [0, cfg.batch_size_train - 1]) + 1) * 255 / 2,
                                                              tf.uint8), global_step, max_image_output)
                    else:
                        for c in range(self.options['in_channels']):
                            tf.summary.image('train_img_c'+str(c), tf.expand_dims(tf.cast(
                                (tf.gather(x[:, :, :, c], [0, cfg.batch_size_train - 1]) + 1) * 255 / 2,
                                tf.uint8), axis=-1), global_step, max_image_output)

                    tf.summary.image('train_seg_lbl', tf.expand_dims(tf.cast(tf.argmax(
                        tf.gather(y, [0, cfg.batch_size_train - 1])
                        , -1) * (255 // (cfg.num_classes_seg - 1)), tf.uint8), axis=-1), global_step, max_image_output)
                    tf.summary.image('train_seg_pred', tf.expand_dims(tf.cast(
                        tf.gather(predictions, [0, cfg.batch_size_train - 1])
                        * (255 // (cfg.num_classes_seg - 1)), tf.uint8), axis=-1), global_step, max_image_output)

                else:
                    if self.options['in_channels'] == 1:
                        tf.summary.image('train_img', tf.cast(
                            (tf.gather(x[:, self.inputs['x'].shape[1] // 2, :, :], [0, cfg.batch_size_train - 1]) + 1) * 255 / 2,
                            tf.uint8), global_step, max_image_output)
                    else:
                        pass

                    tf.summary.image('train_seg_lbl', tf.expand_dims(tf.cast(tf.argmax(
                        tf.gather(y[:, self.inputs['x'].shape[1] // 2], [0, cfg.batch_size_train - 1]),
                        -1) * (255 // (cfg.num_classes_seg - 1)), tf.uint8), axis=-1), global_step, max_image_output)

                    tf.summary.image('train_seg_pred', tf.expand_dims(tf.cast(
                            tf.gather(predictions[:, self.inputs['x'].shape[1] // 2], [0, cfg.batch_size_train - 1])
                            * (255 // (cfg.num_classes_seg - 1)), tf.uint8), axis=-1), global_step, max_image_output)

            with tf.name_scope('02_Probabilities'):
                if self.options['rank'] == 2:
                    for c in range(cfg.num_classes_seg):
                        tf.summary.image('train_seg_prob_' + str(c), tf.expand_dims(tf.cast(
                            tf.gather(probabilities[:, :, :, c], [0, cfg.batch_size_train - 1])
                            * 255, tf.uint8), axis=-1), global_step, max_image_output)
                else:
                    for c in range(cfg.num_classes_seg):
                        tf.summary.image('train_seg_prob_class' + str(c), tf.expand_dims(tf.cast(
                            tf.gather(probabilities[:, self.inputs['x'].shape[1] // 2, :, :, c],
                                      [0, cfg.batch_size_train - 1])
                            * 255, tf.uint8), axis=-1), global_step, max_image_output)

            with tf.name_scope('03_Class_Labels'):
                if self.options['rank'] == 2:
                    pass
                else:
                    for c in range(cfg.num_classes_seg):
                        tf.summary.image('train_seg_lbl' + str(c), tf.expand_dims(tf.cast(
                            tf.gather(y[:, self.inputs['x'].shape[1] // 2, :, :, c], [0, cfg.batch_size_train - 1])
                            * 255, tf.uint8), axis=-1), global_step, max_image_output)

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

        folder, file_number = os.path.split(file_name[0])
        # Use a SimpleITK reader to load the nii images and labels for training
        data_img = sitk.ReadImage(os.path.join(folder, (cfg.sample_file_name_prefix + file_number + '.nii')))
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

        pred_img = image.np_array_to_itk_image(predictions, data_info, cfg.label_background_value,
                                               cfg.adapt_resolution, cfg.target_type_label)
        if cfg.do_connected_component_analysis:
            if cfg.num_classes_seg == 1:
                pred_img = image.extract_largest_connected_component_sitk(pred_img)
            else:
                pred_img_art = image.extract_largest_connected_component_sitk(pred_img == 1)
                pred_img_vein = image.extract_largest_connected_component_sitk(pred_img == 2)
                pred_img = sitk.Mask(pred_img_art, pred_img_vein < 1, 2)

        elif cfg.do_filter_small_components:
            if cfg.num_classes_seg == 1:
                pred_img = image.remove_small_components_sitk(pred_img, cfg.min_number_of_voxels)
            else:
                pred_img_art = image.remove_small_components_sitk(pred_img == 1, cfg.min_number_of_voxels)
                pred_img_vein = image.remove_small_components_sitk(pred_img == 2, cfg.min_number_of_voxels)
                pred_img = sitk.MaskNegated(pred_img_art, pred_img_vein, 2)


        sitk.WriteImage(pred_img,
                        os.path.join(out_path, ('prediction' + '-' + version + '-' + file_number + '.nii')))

