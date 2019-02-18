import tensorflow as tf
import numpy as np
from time import time
import config as cfg
from NetworkBasis.network import Network
import NetworkBasis.loss as Loss
import NetworkBasis.metric as Metric
import os


class SegBasisNet(Network):

    def _set_iterator(self):
        self.options['batch_size'] = cfg.batch_size
        iterator = tf.data.Iterator.from_string_handle(self.variables['handle'], (cfg.dtype, cfg.dtype), (
            np.append([None], cfg.train_input_shape), (np.append([None], cfg.train_label_shape))))
        self.inputs['x'], self.inputs['y'] = iterator.get_next()

    def _set_placeholder(self):
        self.batch_size = cfg.test_size
        x_placeholder = tf.placeholder(tf.float32, np.append([None], cfg.test_data_shape))
        y_placeholder = tf.placeholder(tf.float32, np.append([None], cfg.test_label_shape_seg))
        self.inputs['x'] = x_placeholder
        self.inputs['y'] = y_placeholder

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

        if cfg.VERBOSE:
            print('Predictions: ', self.outputs['predictions'].dtype)
            print('Logits: ', self.outputs['logits'].dtype)
            print('Input: ', self.inputs['x'].dtype)
            print('Label: ', self.inputs['y'].dtype)

    def _get_loss(self):
        '''!
                Returns loss depending on `self.options['loss']``.

                self.options['loss'] should be in {'DICE', 'TVE', 'CEL', 'WCEL'}.
                @returns @b loss : tf.float
                '''
        # Assert that dimensions match!
        assert self.outputs['logits'].shape.as_list()[1:-1] == self.inputs['y'].shape.as_list()[1:-1]

        # Loss
        if self.options['loss'] == 'DICE':
            loss = Loss.dice_loss(self.outputs['probabilities'], self.inputs['y'])

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

        tf.summary.scalar(self.options['loss'], loss, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
        return loss

    def _set_up_training_image_summaries(self):
        with tf.device('/cpu:0'):
            if self.options['in_channels'] > 1:
                tf.summary.image('train_img', tf.cast((tf.gather(self.inputs['x'], [0, cfg.batch_size-1])
                                [:, :, :, ::self.options['in_channels']//2]+1) * 255/2, tf.uint8),
                                 2, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
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
                tf.summary.scalar('OneHotMax-Img', tf.reduce_max(self.y), collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                help_img = self.inputs['y'][:, :, :, 1]
                for i in range(2, self.n_classes):
                    help_img = tf.expand_dims(tf.add(help_img, tf.multiply(self.y[:, :, :, i], i)), -1)
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
            if self.n_channels > 1:
                tf.summary.image('test_img', tf.cast((self.x[:, :, :, ::self.n_channels//2]+1) * 255/2, tf.uint8),
                                 1, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
            else:
                tf.summary.image('test_img', tf.cast(
                    (self.x + 1) * 255 / 2,
                    tf.uint8), 1, collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])

            if self.n_classes == 2:
                tf.summary.image('test_seg_pred', tf.expand_dims(tf.cast(self.outputs['predictions'] * 255, tf.uint8), axis=-1),
                             1)
                tf.summary.image('test_seg_lbl', tf.expand_dims(tf.cast(self.y[:, :, :, 1] * 255, tf.uint8), axis=-1), 1)
                tf.summary.image('test_seg_prob',
                             tf.expand_dims(tf.cast(self.outputs['probabilities'][:, :, :, 1] * 255, tf.uint8), axis=-1), 1)
                tf.summary.image('test_seg_log_object',
                             tf.expand_dims(self.outputs['logits'][:, :, :, 1], axis=-1), 1)
            else:
                tf.summary.scalar('OneHotMax-Img', tf.reduce_max(self.y),
                                  collections=[tf.GraphKeys.SUMMARIES, 'vald_summaries'])
                help_img = self.y[:, :, :, 1]
                for i in range(2, self.n_classes):
                    help_img = tf.expand_dims(tf.add(help_img, tf.multiply(self.y[:, :, :, i], i)), -1)
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

    def _run_test(self, sess, step, version, test_path, feed_dict, test_files, summaries_per_case=cfg.summaries_per_case):

        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(test_path, graph=self.graph)
        eval_file_path = os.path.join(test_path, os.path.basename(test_path) + '-' + version + '-eval.csv')

        with tf.device('/cpu:0'):
            header_row = evaluation.make_csv_file(eval_file_path)

        # perform tests
        for file in test_files:

            with tf.device('/cpu:0'):
                image_samples, label_samples,             data_info, file_number = evaluation.read_nii_for_testing(file)

            s = image_samples.shape
            predictions = []
            probabilities = []
            start_time = time()

            for i in range(0, s[0], self.batch_size):
                    try:
                        image_slice = image_samples[i:i + self.batch_size]
                        label_slice = label_samples[i:i + self.batch_size]
                    except:
                        image_slice = image_samples[i:-1]
                        label_slice = label_samples[i:-1]
                        if len(image_slice.shape) < 4:
                            image_slice = np.expand_dims(image_slice, axis=0)
                            label_slice = np.expand_dims(label_slice, axis=0)

                    if self.options['batch_normalization']:
                        test_dict = {**feed_dict, **{self.options['is_training_tf']: False,
                                                    self.x: image_slice, self.y: label_slice}}
                    else:
                        test_dict = {**feed_dict, **{self.x: image_slice, self.y: label_slice}}

                    # compute and write summaries
                    if i % (s[0]//summaries_per_case) == 0:
                        [summary, pred, prob] = sess.run([summary_op, self.outputs['predictions'], self.outputs['probabilities']],
                                                          feed_dict=test_dict)
                        writer.add_summary(summary, step)
                    else:
                        [pred, prob] = sess.run([self.outputs['predictions'], self.outputs['probabilities']],
                                                feed_dict=test_dict)

                    pred = np.squeeze(pred)
                    prob = np.squeeze(prob)
                    if len(pred.shape) < 3:
                        pred = np.expand_dims(pred, axis=0)
                        prob = np.expand_dims(prob, axis=0)
                    predictions.append(pred)
                    probabilities.append(prob)

                    # manually adapt step in testing
                    step += 1

            end_time = time()
            del image_samples

            with tf.device('/cpu:0'):
                predictions = np.concatenate(predictions)
                probabilities = np.concatenate(probabilities)

                print('  Pred: ', predictions.shape)
                print('  Prob: ', probabilities.shape)
                print('  Labels: ', label_samples.shape)

                result_metrics = {}
                result_metrics['File Number'] = file_number

                elapsed_time = end_time - start_time
                print('  Elapsed Time (Seconds): ', elapsed_time)
                result_metrics['Time'] = elapsed_time

                if cfg.adapt_resolution:
                    # dice on resampled data
                    res_dice = Metric.dice_coefficient_np(probabilities, label_samples)
                    print('  Resampled Dice Coefficient: ', res_dice)
                    result_metrics['Dice (R)'] = res_dice

                del label_samples

                result_metrics = evaluation.evaluate_segmentation_prediction(predictions, result_metrics, data_info, file, test_path, version)

                evaluation.write_metrics_to_csv(eval_file_path, header_row, result_metrics)

    def _run_apply(self, sess, step, version, apply_path, feed_dict, test_files):

        if not os.path.exists(apply_path):
            os.makedirs(apply_path)

        for file in test_files:

            with tf.device('/cpu:0'):
                image_samples, data_info, file_number = evaluation.read_nii_for_application(file)

            s = image_samples.shape
            predictions = []
            start_time = time()

            for i in range(0, s[0], self.batch_size):
                try:
                    image_slice = image_samples[i:i + self.batch_size]
                except:
                    image_slice = image_samples[i:-1]
                    if len(image_slice.shape) < 4:
                        image_slice = np.expand_dims(image_slice, axis=0)

                if self.options['batch_normalization']:
                    test_dict = {**feed_dict, **{self.options['is_training_tf']: False,
                                                 self.x: image_slice}}
                else:
                    test_dict = {**feed_dict, **{self.x: image_slice}}

                [pred] = sess.run([self.outputs['predictions']], feed_dict=test_dict)

                pred = np.squeeze(pred)
                if len(pred.shape) < 3:
                    pred = np.expand_dims(pred, axis=0)
                predictions.append(pred)

                # manually adapt step in testing
                step += 1

            end_time = time()
            del image_samples

            with tf.device('/cpu:0'):
                elapsed_time = end_time - start_time
                print('  Elapsed Time (Seconds): ', elapsed_time)

                predictions = np.concatenate(predictions)
                evaluation.process_and_write_predictions_nii(predictions, data_info, file_number, apply_path, version)
