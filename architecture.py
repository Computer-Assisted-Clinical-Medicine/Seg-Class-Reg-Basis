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

class DenseTiramisu(SegBasisNet):
    '''
    Implements the One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation.
    inspired by https://github.com/SimJeg/FC-DenseNet
    '''

    def __init__(self, loss, is_training=True, kernel_dims=3, growth_rate=16, layers_per_block=(4, 5, 7, 10, 12),
        bottleneck_layers=15, drop_out=(True, 0.2), regularize=(True, 'L2', 0.001), do_batch_normalization=True,
        do_bias=False, activation='relu', **kwargs):
        """Initialize the 100 layers Tiramisu

        Parameters
        ----------
        loss : str
            the type of loss to use
        is_training : bool, optional
            if in training, by default True
        kernel_dims : int, optional
            the dimensions of the kernel (dimension is automatic), by default 3
        growth_rate : int, optional
            the growth rate in the dense blocks, by default 16
        layers_per_block : tuple, optional
            number of layers per block, by default (4, 5, 7, 10, 12) (from paper)
        bottleneck_layers : int, optional
            number of layers in the bottleneck, by default 15 (from paper)
        drop_out : tuple, optional
            if dropout should be used, by default (True, 0.2) (from paper)
        regularize : tuple, optional
            if there should be regularization, by default (True, 'L2', 0.001)
        do_batch_normalization : bool, optional
            has to be true for dense nets, by default True
        do_bias : bool, optional
            has to be false because batch norm, by default False
        activation : str, optional
            which activation should be used, by default 'relu'
        """

        if do_bias:
            print('use no bias with this network, bias set to False')
            do_bias = False
        if not do_batch_normalization:
            print('always uses batch norm, set to True')
            do_batch_normalization = True

        super().__init__(loss, is_training, kernel_dims=kernel_dims, growth_rate=growth_rate,
            layers_per_block=layers_per_block, bottleneck_layers=bottleneck_layers, drop_out=drop_out,
            regularize=regularize, do_bias=False, do_batch_normalization=True,
            activation=activation, **kwargs)

    @staticmethod
    def get_name():
        return 'DenseTiramisu'

    def conv_layer(self, x, filters:int, name:str):
        """
        Forms the atomic layer of the tiramisu, does three operation in sequence:
        batch normalization -> Relu -> 2D Convolution.

        Parameters
        ----------
        x: Tensor
            input feature map.
        filters: int
            indicating the number of filters in the output feat. map.
        name: str
            name of the layer

        Returns
        -------
        Tensor
            Result of applying batch norm -> Relu -> Convolution.
        """
        bn_layer = tf.keras.layers.BatchNormalization(name=name + '/bn')
        x = bn_layer(x)

        activation_layer = tf.keras.layers.Activation(self.options['activation'], name=name + '/act')
        x = activation_layer(x)

        convolutional_layer = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=self.options['kernel_dims'],
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            activation=None,
            use_bias=False,
            kernel_regularizer=self.options['regularizer'],
            name=name + '/conv2d'
        )
        x = convolutional_layer(x)

        if self.options['drop_out'][0]:
            dropout_layer = tf.keras.layers.Dropout(rate=self.options['drop_out'][1], name=name + '/dropout')
            x = dropout_layer(x)

        return x

    def dense_block(self, x, n_layers:int, name:str):
        """
        Forms the dense block of the Tiramisu to calculate features at a specified growth rate.
        Each conv layer in the dense block calculate self.options['growth_rate'] feature maps,
        which are sequentially concatenated to build a larger final output.

        Parameters
        ----------
        x: Tensor
            input to the Dense Block.
        n_layers: int
            the number of layers in the block
        name: str
            name of the layer

        Returns
        -------
        Tensor
            the output of the dense block.
        """
        layer_outputs = []
        for i in range(n_layers):
            conv = self.conv_layer(x, self.options['growth_rate'], name=name + f'/conv{i}')
            layer_outputs.append(conv)
            if i != n_layers - 1:
                concat_layer = tf.keras.layers.Concatenate(axis=3, name=name + f'/concat{i}')
                x = concat_layer([conv, x])

        final_concat_layer = tf.keras.layers.Concatenate(axis=3, name=name + f'/concat_conv')
        x = final_concat_layer(layer_outputs)
        return x

    def transition_down(self, x, filters:int, name:str):
        """
        Down-samples the input feature map by half using maxpooling.

        Parameters
        ----------
        x: Tensor
            input to downsample.
        filters: int
            number of output filters.
        name: str
            name of the layer

        Returns
        -------
        Tensor
            result of downsampling.
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
            use_bias=False,
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

        Parameters
        ----------
        x: Tensor
            input feature map to upsample.
        filters: int
            number of filters in the output.
        name: str
            name of the layer

        Returns
        -------
        Tensor
            result of up-sampling.
        """
        conv_transpose_layer = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=self.options['kernel_dims'],
            strides=(2, 2),
            padding='same',
            use_bias=False,
            kernel_regularizer=self.options['regularizer'],
            name=name + '_trans_up'
        )
        x = conv_transpose_layer(x)

        return x

    def _build_model(self):
        '''!
        Builds DenseTiramisu

        '''
        # TODO: make 3D version
        self.options['name'] = 'DenseTiramisu'
        self.options['n_blocks'] = len(self.options['layers_per_block'])

        x = self.inputs['x']
        if tf.rank(x).numpy() != 4:
            raise NotImplementedError('Network only suitable for 2D')
        logger.debug('Start model definition')
        logger.debug('Input Shape: %s', x.get_shape())

        concats = []

        # encoder
        first_layer = tf.keras.layers.Conv2D(
            filters=48,
            kernel_size=self.options['kernel_dims'],
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            kernel_regularizer=self.options['regularizer'],
            name='encoder/conv2d'
        )
        x = first_layer(x)
        logger.debug('First Convolution Out: %s', x.get_shape())

        for block_nb in range(0, self.options['n_blocks']):
            dense = self.dense_block(x, self.options['layers_per_block'][block_nb], name=f'down_block{block_nb}')
            concat_layer = tf.keras.layers.Concatenate(axis=3, name=f'concat_output_down{block_nb-1}')
            x = concat_layer([x, dense])
            concats.append(x)
            x = self.transition_down(x, x.get_shape()[-1], name=f'transition_down{block_nb}')
            logger.debug('Downsample Out: %s', x.get_shape())
            logger.debug('m=%i', x.get_shape()[-1])

        x = self.dense_block(x, self.options['bottleneck_layers'], name='bottleneck')
        logger.debug('Bottleneck Block: %s', x.get_shape())

        # decoder
        for i, block_nb in enumerate(range(self.options['n_blocks']-1, -1, -1)):
            logger.debug('Block %i', i)
            logger.debug('Block to upsample: %s', x.get_shape())
            x = self.transition_up(x, x.get_shape()[-1], name=f'transition_up{i}')
            logger.debug('Upsample out: %s', x.get_shape())
            concat_layer = tf.keras.layers.Concatenate(axis=3, name=f'concat_input{i}')
            x_con = concat_layer([x, concats[len(concats) - i - 1]])
            logger.debug('Skip connect: %s', concats[len(concats) - i - 1].get_shape())
            logger.debug('Concat out: %s', x_con.get_shape())
            x = self.dense_block(x_con, self.options['layers_per_block'][block_nb], name=f'up_block{i}')
            logger.debug('Dense out: %s', x.get_shape())
            logger.debug('m=%i', x.get_shape()[3] + x_con.get_shape()[3])

        # concatenate the last dense block
        concat_layer = tf.keras.layers.Concatenate(axis=3, name='last_concat')
        x = concat_layer([x, x_con])

        logger.debug('Last layer in: %s', x.get_shape())

        # prediction
        last_layer = tf.keras.layers.Conv2D(
            filters=self.options['out_channels'],
            kernel_size=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            kernel_regularizer=self.options['regularizer'],
            activation=None,
            use_bias=False,
            name='prediction/conv2d'
        )
        x = last_layer(x)

        last_activation_layer = tf.keras.layers.Activation(self._select_final_activation(), name='prediction/act')
        self.outputs['probabilities'] = last_activation_layer(x)
        
        logger.debug('Mask Prediction: %s', x.get_shape())
        logger.debug('Finished model definition.')

        return CustomKerasModel(inputs=self.inputs['x'], outputs=self.outputs['probabilities'])