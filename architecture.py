'''Collection of architecures that can be used for segmentation
'''
import logging

import numpy as np
import tensorflow as tf

from . import config as cfg
from .NetworkBasis import block, layer
from .segbasisnet import SegBasisNet

#configure logger
logger = logging.getLogger(__name__)


CustomKerasModel = tf.keras.models.Model
# class CustomKerasModel(tf.keras.models.Model):
#     '''
#     This can be used to customize the training loop or the other training steps.
#     '''
#     def __init__(self, **kwargs):
#         super(CustomKerasModel, self).__init__(**kwargs)


class UNet(SegBasisNet):
    '''!
    U-Net # TODO: add reference

    %Network Architecture from paper # TODO: revise for implementation

    - **Encoding Path** (4 Encoding Blocks): 4x{
        -  3x3 convolution + ReLU
        -  3x3 convolution + ReLU
        -  2x2 max pooling operation with stride 2 for downsampling
        ( double the number of feature channels)}
    - **Bottom**
        -  3x3 convolution + ReLU
        -  3x3 convolution + ReLU
    - **Decoding Path** (4 Decoding Blocks): 4x{
        -  an upsampling of the feature map followed by a 2x2 up-convolution  # TODO: revise
        -  a concatenation with the correspondingly cropped feature map from the encoding path,
        -  3x3 convolution + ReLU
        -  3x3 convolution + ReLU}
    -  **Final Layer**  to compute logits.
        - 1x1 convolution.
    '''

    def __init__(self, loss, is_training=True, do_finetune=False, model_path="",
                 n_filters=[8, 16, 32, 64, 128], kernel_dims=3, n_convolutions=[2, 3, 2], drop_out=[False, 0.2],
                 regularize=[True, 'L2', 0.00001], do_batch_normalization=False, do_bias=True,
                 activation='relu', upscale='TRANS_CONV', downscale='MAX_POOL', res_connect=False, skip_connect=True,
                 cross_hair=False, **kwargs):
        super(UNet, self).__init__(loss, is_training, do_finetune, model_path,
                 n_filters, kernel_dims, n_convolutions, drop_out,
                 regularize, do_batch_normalization, do_bias,
                 activation, upscale, downscale, res_connect, skip_connect, cross_hair, **kwargs)

    @staticmethod
    def get_name():
        return 'UNet'

    def _build_model(self):
        '''!
        Builds U-Net

        '''
        ## Name of the network
        self.options['name'] = self.get_name()
        self.options['n_filters_per_block'] = [*self.options['n_filters'], *self.options['n_filters'][-2::-1]]
        self.options['n_blocks'] = len(self.options['n_filters_per_block'])

        #self._print_init()

        x = self.inputs['x']

        assert x.shape[1] % 16 == 0, 'N_Slices has to be divisible by 16, otherwise the downsampling fails'

        # Encoding
        for block_index in range(0, 2):
            with tf.name_scope('%02d_enc_block' % (block_index)):
                logger.debug(' Encoding Block %s', block_index)
                x = block.encoding(x, self.options, self.variables, self.options['n_convolutions'][0],
                                   self.options['kernel_dims'],
                                   self.options['n_filters_per_block'][block_index], self.options['strides'],
                                   self.options['padding'], self.options['dilation_rate'],
                                   self.options['activation'], self.options['use_bias'],
                                   self.options['batch_normalization'], self.options['drop_out'])
                # self.variables['feature_maps'].append(x) is performed in the encoding block
                logger.debug(' Result is Tensor with shape %s', x.shape)

        # Encoding
        for block_index in range(2, 4):
            with tf.name_scope('%02d_enc_block' % (block_index)):
                logger.debug(' Encoding Block %s', block_index)
                x = block.encoding(x, self.options, self.variables, self.options['n_convolutions'][1],
                                   self.options['kernel_dims'],
                                   self.options['n_filters_per_block'][block_index], self.options['strides'],
                                   self.options['padding'], self.options['dilation_rate'],
                                   self.options['activation'], self.options['use_bias'],
                                   self.options['batch_normalization'], self.options['drop_out'])
                # self.variables['feature_maps'].append(x) is performed in the encoding block
                logger.debug(' Result is Tensor with shape %s', x.shape)


        # Bottom
        block_index = 4
        with tf.name_scope('%02d_bot_block' % block_index):
            logger.debug(' Bottom Block %s', block_index)
            x = block.basic(x, self.options, self.options['n_convolutions'][1],
                            self.options['kernel_dims'],
                            self.options['n_filters_per_block'][block_index], self.options['strides'],
                            self.options['padding'], self.options['dilation_rate'],
                            self.options['activation'], self.options['use_bias'],
                            self.options['batch_normalization'], self.options['drop_out'])
            logger.debug(' Result is Tensor with shape %s', x.shape)

        # Decoding
        for block_index in range(5, 7):
            with tf.name_scope('%02d_dec_block' % block_index):
                logger.debug(' Decoding Block %s', block_index)
                x = block.decoding(x,
                                   self.options, self.options['n_convolutions'][1],
                                   self.options['kernel_dims'],
                                   self.options['n_filters_per_block'][block_index], self.options['strides'],
                                   self.options['padding'], self.options['dilation_rate'],
                                   self.options['activation'], self.options['use_bias'],
                                   self.options['batch_normalization'], self.options['drop_out'],
                                   self.variables['feature_maps'][self.options['n_blocks'] - block_index - 1])
                logger.debug(' Result is Tensor with shape %s', x.shape)

        # Decoding
        for block_index in range(7, 9):
            with tf.name_scope('%02d_dec_block' % block_index):
                logger.debug(' Decoding Block %s', block_index)
                x = block.decoding(x,
                                   self.options, self.options['n_convolutions'][2],
                                   self.options['kernel_dims'],
                                   self.options['n_filters_per_block'][block_index], self.options['strides'],
                                   self.options['padding'], self.options['dilation_rate'],
                                   self.options['activation'], self.options['use_bias'],
                                   self.options['batch_normalization'], self.options['drop_out'],
                                   self.variables['feature_maps'][self.options['n_blocks'] - block_index - 1])
                logger.debug(' Result is Tensor with shape %s', x.shape)

        # Add final 1x1 convolutional layer to compute logits
        with tf.name_scope('9_last_layer' + str(cfg.num_classes_seg)):
            self.outputs['probabilities'] = layer.last(
                x, self.outputs, np.ones(self.options['rank'], dtype=np.int32),
                self.options['out_channels'], self.options['strides'],
                self.options['padding'], self.options['dilation_rate'],
                self._select_final_activation(), False,
                self.options['regularizer'], self.options['use_cross_hair'],
                do_summary=True
            )
            logger.debug(' Probabilities have shape %s', self.outputs['probabilities'].shape)

        return CustomKerasModel(inputs=self.inputs['x'], outputs=self.outputs['probabilities'])


class DVN(SegBasisNet):
    '''!
    DeepVesselNet # TODO: add reference

    %Network Architecture from paper TODO: revise for implementation

    - **Encoding Path** (4 Encoding Blocks): 4x{
        -  3x3 convolution + ReLU
        -  3x3 convolution + ReLU
        -  2x2 max pooling operation with stride 2 for downsampling
        ( double the number of feature channels)}
    - **Bottom**
        -  3x3 convolution + ReLU
        -  3x3 convolution + ReLU
    - **Decoding Path** (4 Decoding Blocks): 4x{
        -  an upsampling of the feature map followed by a 2x2 up-convolution  TODO: revise
        -  a concatenation with the correspondingly cropped feature map from the encoding path,
        -  3x3 convolution + ReLU
        -  3x3 convolution + ReLU}
    -  **Final Layer**  to compute logits.
        - 1x1 convolution.
    '''

    def __init__(self, loss, is_training=True, do_finetune=False, model_path="",
                 n_filters=[4, 8, 16, 32, 1], kernel_dims=[3, 5, 5, 3], n_convolutions=1, drop_out=[False, 0.2],
                 regularize=[True, 'L2', 0.00001], do_batch_normalization=False, do_bias=True,
                 activation='tanh', upscale=None, downscale=None, res_connect=False, skip_connect=False,
                 cross_hair=True, **kwargs):
        super(DVN, self).__init__(loss, is_training, do_finetune, model_path,
                                      n_filters, kernel_dims, n_convolutions, drop_out,
                                      regularize, do_batch_normalization, do_bias,
                                      activation, upscale, downscale, res_connect, skip_connect, cross_hair, **kwargs)

    @staticmethod
    def get_name():
        return 'DVN'

    def _build_model(self):
        '''!
        Builds FCN

        '''
        ## Name of the network
        self.options['name'] = self.get_name()

        #self._print_init()

        x = self.inputs['x']

        # Convolutional Layers
        for block_index in range(0, 4):
            with tf.name_scope('%02dconv_block' % (block_index)):
                logger.debug(' Convolutional Block %s', block_index)
                x = block.basic(x, self.options, self.options['n_convolutions'],
                                   [self.options['kernel_dims'][0][block_index]] * self.options['rank'],
                                   self.options['n_filters'][block_index], self.options['strides'],
                                   self.options['padding'], self.options['dilation_rate'],
                                   self.options['activation'], self.options['use_bias'],
                                   self.options['batch_normalization'], self.options['drop_out'])

                logger.debug(' Result is Tensor with shape %s', x.shape)

        # Add final 1x1 convolutional layer to compute logits
        with tf.name_scope('4_last_layer'+ str(cfg.num_classes_seg)):
            self.outputs['probabilities'] = layer.last(x, self.outputs,
                                                       np.ones(self.options['rank'], dtype=np.int32),
                                                       self.options['out_channels'],
                                                       self.options['strides'],
                                                       self.options['padding'],
                                                       self.options['dilation_rate'],
                                                       self._select_final_activation(), False,
                                                       self.options['regularizer'],
                                                       self.options['use_cross_hair'], do_summary=True)

            logger.debug(' Probabilities have shape %s', self.outputs['probabilities'].shape)

        return CustomKerasModel(inputs=self.inputs['x'], outputs=self.outputs['probabilities'])


class VNet(SegBasisNet):
    '''!
    U-Net # TODO: add reference

    %Network Architecture from paper TODO: revise for implementation

    - **Encoding Path** (4 Encoding Blocks): 4x{
        -  3x3 convolution + ReLU
        -  3x3 convolution + ReLU
        -  2x2 max pooling operation with stride 2 for downsampling
        ( double the number of feature channels)}
    - **Bottom**
        -  3x3 convolution + ReLU
        -  3x3 convolution + ReLU
    - **Decoding Path** (4 Decoding Blocks): 4x{
        -  an upsampling of the feature map followed by a 2x2 up-convolution  TODO: revise
        -  a concatenation with the correspondingly cropped feature map from the encoding path,
        -  3x3 convolution + ReLU
        -  3x3 convolution + ReLU}
    -  **Final Layer**  to compute logits.
        - 1x1 convolution.
    '''

    def __init__(self, loss, is_training=True, do_finetune=False, model_path="",
                 n_filters=[8, 16, 32, 64, 128], kernel_dims=5, n_convolutions=[1, 2, 3, 2, 1], drop_out=[True, 0.2],
                 regularize=[True, 'L2', 0.00001], do_batch_normalization=True, do_bias=False,
                 activation='leaky_relu', upscale='TRANS_CONV', downscale='STRIDE', res_connect=True, skip_connect=True,
                 cross_hair=False, **kwargs):
        super(VNet, self).__init__(loss, is_training, do_finetune, model_path,
                 n_filters, kernel_dims, n_convolutions, drop_out,
                 regularize, do_batch_normalization, do_bias,
                 activation, upscale, downscale, res_connect, skip_connect, cross_hair, **kwargs)

    @staticmethod
    def get_name():
        return 'VNet'

    def _build_model(self):
        '''!
        Builds V-Net

        '''
        ## Name of the network
        self.options['name'] = self.get_name()
        self.options['n_filters_per_block'] = [*self.options['n_filters'], *self.options['n_filters'][-2::-1]]
        self.options['n_blocks'] = len(self.options['n_filters_per_block'])

        #self._print_init()

        x = self.inputs['x']

        assert x.shape[1] % 16 == 0, 'N_Slices has to be divisible by 16, otherwise the downsampling fails'

        if self.options['in_channels'] == 1:
            x = tf.tile(x, [1, 1, 1, 1, self.options['n_filters_per_block'][0]])
            x = tf.keras.layers.BatchNormalization()(x)

        else:
            x = layer.convolutional(x, np.ones(self.options['rank'], dtype=np.int32),
                    self.options['n_filters_per_block'][0], self.options['strides'],
                    self.options['padding'], self.options['dilation_rate'],
                    self.options['activation'], False, True, [False, 0.0],
                    self.options['regularizer'], self.options['use_cross_hair'],
                    do_summary=True)

        # Encoding
        for block_index in range(0, 1):
            with tf.name_scope('%02d_enc_block' % (block_index)):
                logger.debug(' Encoding Block %s', block_index)
                x = block.encoding_2(x, self.options, self.variables, self.options['n_convolutions'][0],
                                     self.options['kernel_dims'],
                                     self.options['n_filters_per_block'][block_index], self.options['strides'],
                                     self.options['padding'], self.options['dilation_rate'],
                                     self.options['activation'], self.options['use_bias'],
                                     self.options['batch_normalization'], self.options['drop_out'])
                # self.variables['feature_maps'].append(x) is performed in the encoding block

                logger.debug(' Result is Tensor with shape %s', x.shape)

        # Encoding
        for block_index in range(1, 2):
            with tf.name_scope('%02d_enc_block' % (block_index)):
                logger.debug(' Encoding Block %s', block_index)
                x = block.encoding_2(x, self.options, self.variables, self.options['n_convolutions'][1],
                                   self.options['kernel_dims'],
                                   self.options['n_filters_per_block'][block_index], self.options['strides'],
                                   self.options['padding'], self.options['dilation_rate'],
                                   self.options['activation'], self.options['use_bias'],
                                   self.options['batch_normalization'], self.options['drop_out'])
                # self.variables['feature_maps'].append(x) is performed in the encoding block

                logger.debug(' Result is Tensor with shape %s', x.shape)

        # Encoding
        for block_index in range(2, 4):
            with tf.name_scope('%02d_enc_block' % (block_index)):
                logger.debug(' Encoding Block %s', block_index)
                x = block.encoding_2(x, self.options, self.variables, self.options['n_convolutions'][2],
                                   self.options['kernel_dims'],
                                   self.options['n_filters_per_block'][block_index], self.options['strides'],
                                   self.options['padding'], self.options['dilation_rate'],
                                   self.options['activation'], self.options['use_bias'],
                                   self.options['batch_normalization'], self.options['drop_out'])
                # self.variables['feature_maps'].append(x) is performed in the encoding block

                logger.debug(' Result is Tensor with shape %s', x.shape)


        # Bottom
        block_index = 4
        with tf.name_scope('%02d_bot_block' % block_index):
            logger.debug(' Bottom Block %s', block_index)
            x = block.basic_2(x, self.options, self.options['n_convolutions'][2],
                            self.options['kernel_dims'],
                            self.options['n_filters_per_block'][block_index], self.options['strides'],
                            self.options['padding'], self.options['dilation_rate'],
                            self.options['activation'], self.options['use_bias'],
                            self.options['batch_normalization'], self.options['drop_out'])

            logger.debug(' Result is Tensor with shape %s', x.shape)

        # Decoding
        for block_index in range(5, 7):
            with tf.name_scope('%02d_dec_block' % block_index):
                logger.debug(' Decoding Block %s', block_index)
                x = block.decoding_2(x,
                                   self.options, self.options['n_convolutions'][2],
                                   self.options['kernel_dims'],
                                   self.options['n_filters_per_block'][block_index], self.options['strides'],
                                   self.options['padding'], self.options['dilation_rate'],
                                   self.options['activation'], self.options['use_bias'],
                                   self.options['batch_normalization'], self.options['drop_out'],
                                   self.variables['feature_maps'][self.options['n_blocks'] - block_index - 1])

                logger.debug(' Result is Tensor with shape %s', x.shape)

        # Decoding
        for block_index in range(7, 8):
            with tf.name_scope('%02d_dec_block' % block_index):
                logger.debug(' Decoding Block %s', block_index)
                x = block.decoding_2(x,
                                   self.options, self.options['n_convolutions'][1],
                                   self.options['kernel_dims'],
                                   self.options['n_filters_per_block'][block_index], self.options['strides'],
                                   self.options['padding'], self.options['dilation_rate'],
                                   self.options['activation'], self.options['use_bias'],
                                   self.options['batch_normalization'], self.options['drop_out'],
                                   self.variables['feature_maps'][self.options['n_blocks'] - block_index - 1])

                logger.debug(' Result is Tensor with shape %s', x.shape)

        # Decoding
        for block_index in range(8, 9):
            with tf.name_scope('%02d_dec_block' % block_index):
                logger.debug(' Decoding Block %s', block_index)
                x = block.decoding_2(x,
                                     self.options, self.options['n_convolutions'][0],
                                     self.options['kernel_dims'],
                                     self.options['n_filters_per_block'][block_index], self.options['strides'],
                                     self.options['padding'], self.options['dilation_rate'],
                                     self.options['activation'], self.options['use_bias'],
                                     self.options['batch_normalization'], self.options['drop_out'],
                                     self.variables['feature_maps'][self.options['n_blocks'] - block_index - 1])

                logger.debug(' Result is Tensor with shape %s', x.shape)

        # Add final 1x1 convolutional layer to compute logits
        with tf.name_scope('9_last_layer'+ str(cfg.num_classes_seg)):
            self.outputs['probabilities'] = layer.last(
                x, self.outputs, np.ones(self.options['rank'], dtype=np.int32),
                self.options['out_channels'], self.options['strides'],
                self.options['padding'], self.options['dilation_rate'],
                self._select_final_activation(), False,
                self.options['regularizer'], self.options['use_cross_hair'],
                do_summary=True
            )

            logger.debug(' Probabilities have shape %s', self.outputs['probabilities'].shape)

        return CustomKerasModel(inputs=self.inputs['x'], outputs=self.outputs['probabilities'])


class ResNet(SegBasisNet):
    '''!
    U-Net # TODO: add reference

    %Network Architecture from paper # TODO: revise for implementation

    - **Encoding Path** (4 Encoding Blocks): 4x{
        -  3x3 convolution + ReLU
        -  3x3 convolution + ReLU
        -  2x2 max pooling operation with stride 2 for downsampling
        ( double the number of feature channels)}
    - **Bottom**
        -  3x3 convolution + ReLU
        -  3x3 convolution + ReLU
    - **Decoding Path** (4 Decoding Blocks): 4x{
        -  an upsampling of the feature map followed by a 2x2 up-convolution  # TODO: revise
        -  a concatenation with the correspondingly cropped feature map from the encoding path,
        -  3x3 convolution + ReLU
        -  3x3 convolution + ReLU}
    -  **Final Layer**  to compute logits.
        - 1x1 convolution.
    '''
    def __init__(self, loss, is_training=True, do_finetune=False, model_path="",
                 n_filters=[64, 128, 256, 512, 1024], kernel_dims=3, n_convolutions=[2, 3, 2], drop_out=[False, 0.2],
                 regularize=[True, 'L2', 0.00001], do_batch_normalization=False, do_bias=True,
                 activation='relu', upscale='TRANS_CONV', downscale='MAX_POOL', res_connect=True, skip_connect=False,
                 cross_hair=False, **kwargs):
        super(ResNet, self).__init__(loss, is_training, do_finetune, model_path,
                 n_filters, kernel_dims, n_convolutions, drop_out,
                 regularize, do_batch_normalization, do_bias,
                 activation, upscale, downscale, res_connect, skip_connect, cross_hair, **kwargs)

    @staticmethod
    def get_name():
        return 'ResNet'

    def _build_model(self):
        '''!
        Builds U-Net

        '''
        ## Name of the network
        self.options['name'] = 'ResNet'
        self.options['n_filters_per_block'] = [*self.options['n_filters'], *self.options['n_filters'][-2::-1]]
        self.options['n_blocks'] = len(self.options['n_filters_per_block'])
        self.options['batch_normalization_per_block'] = [self.options['batch_normalization']] * self.options['n_blocks']
        self.options['activation_per_block'] = [self.options['activation']] * self.options['n_blocks']
        self.options['padding_per_block'] = [self.options['padding']] * self.options['n_blocks']
        self.options['kernel_dims_per_block'] = [self.options['kernel_dims']] * self.options['n_blocks']

        #self._print_init()

        x = self.inputs['x']

        # Encoding
        for block_index in range(0, 2):
            with tf.name_scope('%02d_enc_block' % (block_index)):
                logger.debug(' Encoding Block %s', block_index)
                x = block.encoding(x, self.options, self.variables, self.options['n_convolutions'][0],
                                   self.options['kernel_dims_per_block'][block_index],
                                   self.options['n_filters_per_block'][block_index], self.options['dilation_rate'],
                                   self.options['padding_per_block'][block_index], self.options['dilation_rate'],
                                   self.options['activation_per_block'][block_index], self.options['use_bias'],
                                   self.options['batch_normalization_per_block'][block_index], self.options['drop_out'])
                # self.variables['feature_maps'].append(x) is performed in the encoding block
                logger.debug(' Result is Tensor with shape %s', x.shape)

        # Encoding
        for block_index in range(2, 4):
            with tf.name_scope('%02d_enc_block' % (block_index)):
                logger.debug(' Encoding Block %s', block_index)
                x = block.encoding(x, self.options, self.variables, self.options['n_convolutions'][1],
                                   self.options['kernel_dims_per_block'][block_index],
                                   self.options['n_filters_per_block'][block_index], self.options['dilation_rate'],
                                   self.options['padding_per_block'][block_index], self.options['dilation_rate'],
                                   self.options['activation_per_block'][block_index], self.options['use_bias'],
                                   self.options['batch_normalization_per_block'][block_index], self.options['drop_out'])
                # self.variables['feature_maps'].append(x) is performed in the encoding block
                logger.debug(' Result is Tensor with shape %s', x.shape)

        # Bottom
        block_index = 4
        with tf.name_scope('%02d_bot_block' % block_index):
            logger.debug(' Bottom Block %s', block_index)
            x = block.basic(x, self.options, self.options['n_convolutions'][1],
                            self.options['kernel_dims_per_block'][block_index],
                            self.options['n_filters_per_block'][block_index], self.options['dilation_rate'],
                            self.options['padding_per_block'][block_index], self.options['dilation_rate'],
                            self.options['activation_per_block'][block_index], self.options['use_bias'],
                            self.options['batch_normalization_per_block'][block_index], self.options['drop_out'])

            logger.debug(' Result is Tensor with shape %s', x.shape)

        # Decoding
        for block_index in range(5, 7):
            with tf.name_scope('%02d_dec_block' % block_index):
                logger.debug(' Decoding Block %s', block_index)
                x = block.decoding(x,
                                 self.options, self.options['n_convolutions'][1],
                                 self.options['kernel_dims_per_block'][block_index],
                                 self.options['n_filters_per_block'][block_index], self.options['dilation_rate'],
                                 self.options['padding_per_block'][block_index], self.options['dilation_rate'],
                                 self.options['activation_per_block'][block_index], self.options['use_bias'],
                                 self.options['batch_normalization_per_block'][block_index], self.options['drop_out'])

                logger.debug(' Result is Tensor with shape %s', x.shape)

        # Decoding
        for block_index in range(7, 9):
            with tf.name_scope('%02d_dec_block' % block_index):
                logger.debug(' Decoding Block %s', block_index)
                x = block.decoding(x,
                                   self.options, self.options['n_convolutions'][2],
                                   self.options['kernel_dims_per_block'][block_index],
                                   self.options['n_filters_per_block'][block_index], self.options['dilation_rate'],
                                   self.options['padding_per_block'][block_index], self.options['dilation_rate'],
                                   self.options['activation_per_block'][block_index], self.options['use_bias'],
                                   self.options['batch_normalization_per_block'][block_index], self.options['drop_out'])
                    
                logger.debug(' Result is Tensor with shape %s', x.shape)

        # Add final 1x1 convolutional layer to compute logits
        with tf.name_scope('9_last_layer'+ str(cfg.num_classes_seg)):
            self.outputs['probabilities'] = layer.last(
                x=x,
                outputs=self.outputs,
                filter_shape=np.ones(self.options['rank'], dtype=np.int32),
                n_filter=self.options['out_channels'],
                stride=self.options['strides'],
                padding=self.options['padding'],
                dilation_rate=self.options['dilation_rate'],
                act_func=self._select_final_activation(),
                use_bias=False,
                regularizer=self.options['regularizer'],
                cross_hair=self.options['use_cross_hair'],
                do_summary=True
            )
  
            logger.debug(' Probabilities have shape %s', self.outputs['probabilities'].shape)

        return CustomKerasModel(inputs=self.inputs['x'], outputs=self.outputs['probabilities'])

class DenseTiramisu_keras(SegBasisNet):

    def __init__(self, loss, is_training=True, do_finetune=False, model_path="",
                 n_filters=(8, 16, 32, 64, 128), kernel_dims=3, n_convolutions=(2, 3, 2), drop_out=(False, 0.2),
                 regularize=(True, 'L2', 0.00001), do_batch_normalization=False, do_bias=True,
                 activation='relu', upscale='TRANS_CONV', downscale='MAX_POOL', res_connect=False, skip_connect=True,
                 cross_hair=False, **kwargs):

        self.growth_k = 16
        self.layers_per_block = (2, 3, 3)
        self.nb_blocks = len(self.layers_per_block)

        super().__init__(loss, is_training, do_finetune, model_path,
                 n_filters, kernel_dims, n_convolutions, drop_out,
                 regularize, do_batch_normalization, do_bias,
                 activation, upscale, downscale, res_connect, skip_connect, cross_hair, **kwargs)

    @staticmethod
    def get_name():
        return 'DenseTiramisu'

    def dense_block(self, x, blocks, name):
        """A dense block.
        Args:
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.
        Returns:
            Output tensor for the block.
        """
        for i in range(blocks):
            x = self.conv_block(x, 32, name=name + '/block' + str(i + 1))
        return x

    def conv_block(self, x, growth_rate, name):
        """A building block for a dense block.
        Args:
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.
        Returns:
            Output tensor for the block.
        """
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        x_res = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '/0_bn'
        )(x)
        x_res = tf.keras.layers.Activation('relu', name=name + '/0_relu')(x_res)
        x_res = tf.keras.layers.Conv2D(
            4 * growth_rate, 1, use_bias=False, name=name + '/1_conv'
        )(x_res)
        x_res = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '/1_bn'
        )(x_res)
        x_res = tf.keras.layers.Activation('relu', name=name + '/1_relu')(x_res)
        x_res = tf.keras.layers.Conv2D(
            growth_rate, 3, padding='same', use_bias=False, name=name + '/2_conv'
        )(x_res)
        x = tf.keras.layers.Concatenate(axis=bn_axis, name=name + '/concat')([x, x_res])
        return x

    def transition_down(self, x, reduction, name):
        """A transition block.
        Args:
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.
        Returns:
            output tensor for the block.
        """
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '/bn'
        )(x)
        x = tf.keras.layers.Activation('relu', name=name + '/relu')(x)
        x = tf.keras.layers.Conv2D(
            int(tf.keras.backend.int_shape(x)[bn_axis] * reduction),
            1, use_bias=False, name=name + '/conv'
        )(x)
        x = tf.keras.layers.AveragePooling2D(2, strides=2, name=name + '/pool')(x)
        return x

    def transition_up(self, x, reduction, name):
        """
        Up-samples the input feature maps using transpose convolutions.
        Args:
            x: Tensor, input feature map to upsample.
            filters: Integer, number of filters in the output.
        Returns:
            x: Tensor, result of up-sampling.
        """
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        conv_transpose_layer = tf.keras.layers.Conv2DTranspose(
            filters=int(tf.keras.backend.int_shape(x)[bn_axis] * reduction),
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            kernel_regularizer=self.options['regularizer'],
            name=name + '/conv_trans'
        )
        x = conv_transpose_layer(x)

        return x

    def _build_model(self):
        '''!
        Builds DenseTiramisu

        '''
        # TODO: debug
        # TODO: test on simple set

        self.options['name'] = 'DenseTiramisu'
        self.options['n_blocks'] = self.nb_blocks

        img_input = self.inputs['x']

        concats = []

        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

        x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
        x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(
                x)
        x = tf.keras.layers.Activation('relu', name='conv1/relu')(x)
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1')(x)

        for block_nb in range(0, self.nb_blocks):
            x = self.dense_block(x, self.layers_per_block[0], name=f'block{block_nb+1}')
            concats.append(x)
            x = self.transition_down(x, 0.5, name=f'trans{block_nb+1}')
            print("Downsample Out:", x.get_shape())

        print("Bottleneck Block: ", x.get_shape())

        # decoder
        for i, block_nb in enumerate(range(self.nb_blocks - 1, 0, -1)):
            x = self.transition_up(x, 2, name=f'up{block_nb}')
            concat_layer = tf.keras.layers.Concatenate(axis=3, name=f'concatup{block_nb}')
            x = concat_layer([x, concats[len(concats) - i - 1]])
            print("Upsample after concat: ", x.get_shape())
            x = self.dense_block(x, block_nb, name=f'convup{block_nb}')

        # prediction
        with tf.name_scope('prediction'):#, tf.python.keras.backend.get_graph().as_default():
            last_layer = tf.keras.layers.Conv2D(
                filters=self.options['out_channels'],
                kernel_size=(1, 1),
                padding='same',
                dilation_rate=(1, 1),
                kernel_regularizer=self.options['regularizer'],
                activation=None,
                name='last_conv'
            )
            x = last_layer(x)

            last_activation_layer = tf.keras.layers.Activation(self._select_final_activation(), name='last_activation')
            self.outputs['probabilities'] = last_activation_layer(x)
        
        print("Mask Prediction: ", x.get_shape())

        return CustomKerasModel(inputs=self.inputs['x'], outputs=self.outputs['probabilities'])

class DenseTiramisu(SegBasisNet):
# inspired by https://github.com/HasnainRaz/FC-DenseNet-TensorFlow/blob/master/model.py

    def __init__(self, loss, is_training=True, do_finetune=False, model_path="",
                 n_filters=(8, 16, 32, 64, 128), kernel_dims=3, n_convolutions=(2, 3, 2), drop_out=(False, 0.2),
                 regularize=(True, 'L2', 0.00001), do_batch_normalization=False, do_bias=True,
                 activation='relu', upscale='TRANS_CONV', downscale='MAX_POOL', res_connect=False, skip_connect=True,
                 cross_hair=False, **kwargs):

        self.growth_k = 16
        self.layers_per_block = (2, 3, 3)
        self.nb_blocks = len(self.layers_per_block)

        super().__init__(loss, is_training, do_finetune, model_path,
                 n_filters, kernel_dims, n_convolutions, drop_out,
                 regularize, do_batch_normalization, do_bias,
                 activation, upscale, downscale, res_connect, skip_connect, cross_hair, **kwargs)

    @staticmethod
    def get_name():
        return 'DenseTiramisu'

    def conv_layer(self, x, filters:int, name:str):
        """
        Forms the atomic layer of the tiramisu, does three operation in sequence:
        batch normalization -> Relu -> 2D Convolution.
        Args:
            x: Tensor, input feature map.
            filters: Integer, indicating the number of filters in the output feat. map.
            name: str, name of the layer
        Returns:
            x: Tensor, Result of applying batch norm -> Relu -> Convolution.
        """
        bn_layer = tf.keras.layers.BatchNormalization(name=name + '/bn')
        x = bn_layer(x)

        activation_layer = tf.keras.layers.Activation(self.options['activation'])
        x = activation_layer(x)

        convolutional_layer = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            activation=None,
            kernel_regularizer=self.options['regularizer'],
            name=name + '/conv2d'
        )
        x = convolutional_layer(x)

        if self.options['drop_out'][0]:
            dropout_layer = tf.keras.layers.Dropout(rate=self.options['drop_out'][1], name=name + '/dropout')
            x = dropout_layer(x)

        return x

    def dense_block(self, x, block_nb:int, name:str):
        """
        Forms the dense block of the Tiramisu to calculate features at a specified growth rate.
        Each conv layer in the dense block calculate growth_k feature maps, which are sequentially
        concatenated to build a larger final output.
        Args:
            x: Tensor, input to the Dense Block.
            block_nb: Int, identifying the block in the graph.
            name: str, name of the layer
        Returns:
            x: Tensor, the output of the dense block.
        """
        for i in range(self.layers_per_block[block_nb]):
            conv = self.conv_layer(x, self.growth_k, name=name + f'/conv{i}')
            concat_layer = tf.keras.layers.Concatenate(axis=3, name=name + f'/concat{i}')
            x = concat_layer([conv, x])

        return x

    def transition_down(self, x, filters:int, name:str):
        """
        Down-samples the input feature map by half using maxpooling.
        Args:
            x: Tensor, input to downsample.
            filters: Integer, indicating the number of output filters.
            name: str, name of the layer
        Returns:
            x: Tensor, result of downsampling.
        """
        bn_layer = tf.keras.layers.BatchNormalization(name=name + '/bn')
        x = bn_layer(x)

        activation_layer = tf.keras.layers.Activation(self.options['activation'], name=name + '/act')
        x = activation_layer(x)

        convolutional_layer = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            activation=None,
            kernel_regularizer=self.options['regularizer'],
            name=name + '/conv2d'
        )
        x = convolutional_layer(x)

        if self.options['drop_out'][0]:
            dropout_layer = tf.keras.layers.Dropout(rate=self.options['drop_out'][1], name=name + '/dropout')
            x = dropout_layer(x)

        pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=name + '/maxpool')
        x = pooling_layer(x)

        return x

    def transition_up(self, x, filters:int, name:str):
        """
        Up-samples the input feature maps using transpose convolutions.
        Args:
            x: Tensor, input feature map to upsample.
            filters: Integer, number of filters in the output.
            name: str, name of the layer
        Returns:
            x: Tensor, result of up-sampling.
        """
        with tf.name_scope('transition_up'):#, tf.python.keras.backend.get_graph().as_default():
            conv_transpose_layer = tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='same',
                kernel_regularizer=self.options['regularizer'],
                name=name + '_trans_up'
            )
            x = conv_transpose_layer(x)

        return x

    def _build_model(self):
        '''!
        Builds DenseTiramisu

        '''
        # TODO: debug
        # TODO: test on simple set

        self.options['name'] = 'DenseTiramisu'
        self.options['n_blocks'] = self.nb_blocks

        x = self.inputs['x']

        concats = []

        # encoder
        first_layer = tf.keras.layers.Conv2D(
            filters=48,
            kernel_size=(3,3),
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            kernel_regularizer=self.options['regularizer'],
            name='encoder/conv2d'
        )
        x = first_layer(x)
        tf.print("First Convolution Out: ", x.get_shape())

        for block_nb in range(0, self.nb_blocks):
            dense = self.dense_block(x, block_nb, name=f'down_block{block_nb}')

            if block_nb != self.nb_blocks - 1:
                concat_layer = tf.keras.layers.Concatenate(axis=3, name=f'concat_output_down{block_nb-1}')
                x = concat_layer([x, dense])
                concats.append(x)
                x = self.transition_down(x, x.get_shape()[-1], name=f'transition_down{block_nb}')
                print("Downsample Out:", x.get_shape())

        x = dense
        print("Bottleneck Block: ", dense.get_shape())

        # decoder
        for i, block_nb in enumerate(range(self.nb_blocks - 1, 0, -1)):
            x = self.transition_up(x, x.get_shape()[-1], name=f'transition_up{block_nb}')
            concat_layer = tf.keras.layers.Concatenate(axis=3, name=f'concat_input{block_nb}')
            x = concat_layer([x, concats[len(concats) - i - 1]])
            print("Upsample after concat: ", x.get_shape())
            x = self.dense_block(x, block_nb, name=f'up_block{block_nb}')

        # prediction
        last_layer = tf.keras.layers.Conv2D(
            filters=self.options['out_channels'],
            kernel_size=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            kernel_regularizer=self.options['regularizer'],
            activation=None,
            name='prediction/conv2d'
        )
        x = last_layer(x)

        last_activation_layer = tf.keras.layers.Activation(self._select_final_activation(), name='prediction/act')
        self.outputs['probabilities'] = last_activation_layer(x)
        
        print("Mask Prediction: ", x.get_shape())

        return CustomKerasModel(inputs=self.inputs['x'], outputs=self.outputs['probabilities'])

'''
class DenseTiramisu_original(object):
    """
    This class forms the Tiramisu model for segmentation of input images.
    """
    def __init__(self, growth_k, layers_per_block, num_classes):
        """
        Initializes the Tiramisu based on the specified parameters.
        Args:
            growth_k: Integer, growth rate of the Tiramisu.
            layers_per_block: List of integers, the number of layers in each dense block.
            num_classes: Integer: Number of classes to segment.
        """
        self.growth_k = growth_k
        self.layers_per_block = layers_per_block
        self.nb_blocks = len(layers_per_block)
        self.num_classes = num_classes
        self.logits = None

    def xentropy_loss(self, logits, labels):
        """
        Calculates the cross-entropy loss over each pixel in the ground truth
        and the prediction.
        Args:
            logits: Tensor, raw unscaled predictions from the network.
            labels: Tensor, the ground truth segmentation mask.
        Returns:
            loss: The cross entropy loss over each image in the batch.
        """
        labels = tf.cast(labels, tf.int32)
        logits = tf.reshape(logits, [tf.shape(logits)[0], -1, self.num_classes])
        labels = tf.reshape(labels, [tf.shape(labels)[0], -1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name="loss")

        return loss

    def calculate_iou(self, mask, prediction):
        """
        Calculates the mean intersection over union (mean pixel accuracy)
        Args:
            mask: Tensor, The ground truth input segmentation mask.
            prediction: Tensor, the raw unscaled prediction from the network.
        Returns:
            iou: Tensor, average iou over the batch.
            update_op: Tensor op, update operation for the iou metric.
        """
        mask = tf.reshape(tf.one_hot(tf.squeeze(mask), depth=self.num_classes), [
            tf.shape(mask)[0], -1, self.num_classes])
        prediction = tf.reshape(
            prediction, shape=[tf.shape(prediction)[0], -1, self.num_classes])
        iou, update_op = tf.metrics.mean_iou(
            tf.argmax(prediction, 2), tf.argmax(mask, 2), self.num_classes)

        return iou, update_op

    @staticmethod
    def batch_norm(x, training, name):
        """
        Wrapper for batch normalization in tensorflow, updates moving batch statistics
        if training, uses trained parameters if inferring.
        Args:
            x: Tensor, the input to normalize.
            training: Boolean tensor, indicates if training or not.
            name: String, name of the op in the graph.
        Returns:
            x: Batch normalized input.
        """
        with tf.variable_scope(name):
            x = tf.cond(training, lambda: tf.contrib.layers.batch_norm(x, is_training=True, scope=name+'_batch_norm'),
                        lambda: tf.contrib.layers.batch_norm(x, is_training=False, scope=name+'_batch_norm', reuse=True))
        return x

    def conv_layer(self, x, training, filters, name):
        """
        Forms the atomic layer of the tiramisu, does three operation in sequence:
        batch normalization -> Relu -> 2D Convolution.
        Args:
            x: Tensor, input feature map.
            training: Bool Tensor, indicating whether training or not.
            filters: Integer, indicating the number of filters in the output feat. map.
            name: String, naming the op in the graph.
        Returns:
            x: Tensor, Result of applying batch norm -> Relu -> Convolution.
        """
        with tf.name_scope(name):
            x = self.batch_norm(x, training, name=name+'_bn')
            x = tf.nn.relu(x, name=name+'_relu')
            x = tf.layers.conv2d(x,
                                 filters=filters,
                                 kernel_size=[3, 3],
                                 strides=[1, 1],
                                 padding='SAME',
                                 dilation_rate=[1, 1],
                                 activation=None,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 name=name+'_conv3x3')
            x = tf.layers.dropout(x, rate=0.2, training=training, name=name+'_dropout')

        return x

    def dense_block(self, x, training, block_nb, name):
        """
        Forms the dense block of the Tiramisu to calculate features at a specified growth rate.
        Each conv layer in the dense block calculate growth_k feature maps, which are sequentially
        concatenated to build a larger final output.
        Args:
            x: Tensor, input to the Dense Block.
            training: Bool Tensor, indicating whether training or testing.
            block_nb: Int, identifying the block in the graph.
            name: String, identifying the layers in the graph.
        Returns:
            x: Tensor, the output of the dense block.
        """
        dense_out = []
        with tf.name_scope(name):
            for i in range(self.layers_per_block[block_nb]):
                conv = self.conv_layer(x, training, self.growth_k, name=name+'_layer_'+str(i))
                x = tf.concat([conv, x], axis=3)
                dense_out.append(conv)

            x = tf.concat(dense_out, axis=3)

        return x

    def transition_down(self, x, training, filters, name):
        """
        Down-samples the input feature map by half using maxpooling.
        Args:
            x: Tensor, input to downsample.
            training: Bool tensor, indicating whether training or inferring.
            filters: Integer, indicating the number of output filters.
            name: String, identifying the ops in the graph.
        Returns:
            x: Tensor, result of downsampling.
        """
        with tf.name_scope(name):
            x = self.batch_norm(x, training, name=name+'_bn')
            x = tf.nn.relu(x, name=name+'relu')
            x = tf.layers.conv2d(x,
                                 filters=filters,
                                 kernel_size=[1, 1],
                                 strides=[1, 1],
                                 padding='SAME',
                                 dilation_rate=[1, 1],
                                 activation=None,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 name=name+'_conv1x1')
            x = tf.layers.dropout(x, rate=0.2, training=training, name=name+'_dropout')
            x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name=name+'_maxpool2x2')

        return x

    def transition_up(self, x, filters, name):
        """
        Up-samples the input feature maps using transpose convolutions.
        Args:
            x: Tensor, input feature map to upsample.
            filters: Integer, number of filters in the output.
            name: String, identifying the op in the graph.
        Returns:
            x: Tensor, result of up-sampling.
        """
        with tf.name_scope(name):
            x = tf.layers.conv2d_transpose(x,
                                           filters=filters,
                                           kernel_size=[3, 3],
                                           strides=[2, 2],
                                           padding='SAME',
                                           activation=None,
                                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                           name=name+'_trans_conv3x3')

        return x

    def model(self, x, training):
        """
        Defines the complete graph model for the Tiramisu based on the provided
        parameters.
        Args:
            x: Tensor, input image to segment.
            training: Bool Tensor, indicating whether training or not.
        Returns:
            x: Tensor, raw unscaled logits of predicted segmentation.
        """
        concats = []
        with tf.variable_scope('encoder'):
            x = tf.layers.conv2d(x,
                                filters=48,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='SAME',
                                dilation_rate=[1, 1],
                                activation=None,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                name='first_conv3x3')
            print("First Convolution Out: ", x.get_shape())
            for block_nb in range(0, self.nb_blocks):
                dense = self.dense_block(x, training, block_nb, 'down_dense_block_' + str(block_nb))

                if block_nb != self.nb_blocks - 1:
                    x = tf.concat([x, dense], axis=3, name='down_concat_' + str(block_nb))
                    concats.append(x)
                    x = self.transition_down(x, training, x.get_shape()[-1], 'trans_down_' + str(block_nb))
                    print("Downsample Out:", x.get_shape())

            x = dense
            print("Bottleneck Block: ", dense.get_shape())

        with tf.variable_scope('decoder'):
            for i, block_nb in enumerate(range(self.nb_blocks - 1, 0, -1)):
                x = self.transition_up(x, x.get_shape()[-1], 'trans_up_' + str(block_nb))
                x = tf.concat([x, concats[len(concats) - i - 1]], axis=3, name='up_concat_' + str(block_nb))
                print("Upsample after concat: ", x.get_shape())
                x = self.dense_block(x, training, block_nb, 'up_dense_block_' + str(block_nb))

        with tf.variable_scope('prediction'):
            x = tf.layers.conv2d(x,
                                filters=self.num_classes,
                                kernel_size=[1, 1],
                                strides=[1, 1],
                                padding='SAME',
                                dilation_rate=[1, 1],
                                activation=None,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                name='last_conv1x1')
            print("Mask Prediction: ", x.get_shape())

        return x'''
