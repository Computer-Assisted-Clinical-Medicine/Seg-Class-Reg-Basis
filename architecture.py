import tensorflow as tf
import numpy as np

from .NetworkBasis import block as block
from .NetworkBasis import layer as layer
from .segbasisnet import SegBasisNet
from . import config as cfg


class UNet(SegBasisNet):
    '''!
    U-Net TODO: add reference

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
                 n_filters=[64, 128, 256, 512, 1024], kernel_dims=3, n_convolutions=[2, 3, 2], drop_out=[False, 0.8],
                 regularize=[True, 'L2', 0.00001], do_batch_normalization=False, do_bias=True,
                 activation='relu', upscale='TRANS_CONV', downscale='MAX_POOL', res_connect=False, skip_connect=True,
                 cross_hair=False):
            super(UNet, self).__init__(loss, is_training, do_finetune, model_path,
                 n_filters, kernel_dims, n_convolutions, drop_out,
                 regularize, do_batch_normalization, do_bias,
                 activation, upscale, downscale, res_connect, skip_connect, cross_hair)

    def _build_model(self):
        '''!
        Builds U-Net

        '''
        ## Name of the network
        self.options['name'] = 'UNet'
        self.options['n_filters_per_block'] = [*self.options['n_filters'], *self.options['n_filters'][-2::-1]]
        self.options['n_blocks'] = len(self.options['n_filters_per_block'])

        if cfg.VERBOSE:
            self._print_init()

        x = self.inputs['x']

        # Encoding
        for block_index in range(0, 2):
            with tf.name_scope('%02d_enc_block' % (block_index)):
                if cfg.VERBOSE: print(' Encoding Block ', block_index)
                x = block.encoding(x, self.options, self.variables, self.options['n_convolutions'][0],
                                   self.options['kernel_dims'],
                                   self.options['n_filters_per_block'][block_index], self.options['strides'],
                                   self.options['padding'], self.options['dilation_rate'],
                                   self.options['activation'], self.options['use_bias'],
                                   self.options['batch_normalization'], self.options['drop_out'])
                # self.variables['feature_maps'].append(x) is performed in the encoding block
                if cfg.VERBOSE:
                    print(' Result is Tensor with shape ', x.shape)
                    print(' -------------------------------------')

        # Encoding
        for block_index in range(2, 4):
            with tf.name_scope('%02d_enc_block' % (block_index)):
                if cfg.VERBOSE: print(' Encoding Block ', block_index)
                x = block.encoding(x, self.options, self.variables, self.options['n_convolutions'][1],
                                   self.options['kernel_dims'],
                                   self.options['n_filters_per_block'][block_index], self.options['strides'],
                                   self.options['padding'], self.options['dilation_rate'],
                                   self.options['activation'], self.options['use_bias'],
                                   self.options['batch_normalization'], self.options['drop_out'])
                # self.variables['feature_maps'].append(x) is performed in the encoding block
                if cfg.VERBOSE:
                    print(' Result is Tensor with shape ', x.shape)
                    print(' -------------------------------------')


        # Bottom
        block_index = 4
        with tf.name_scope('%02d_bot_block' % block_index):
            if cfg.VERBOSE: print(' Bottom Block ', block_index)
            x = block.basic(x, self.options, self.options['n_convolutions'][1],
                            self.options['kernel_dims'],
                            self.options['n_filters_per_block'][block_index], self.options['strides'],
                            self.options['padding'], self.options['dilation_rate'],
                            self.options['activation'], self.options['use_bias'],
                            self.options['batch_normalization'], self.options['drop_out'])
            if cfg.VERBOSE:
                print(' Result is Tensor with shape ', x.shape)
                print(' -------------------------------------')

        # Decoding
        for block_index in range(5, 7):
            with tf.name_scope('%02d_dec_block' % block_index):
                if cfg.VERBOSE: print(' Decoding Block ', block_index)
                x = block.decoding(x,
                                   self.options, self.options['n_convolutions'][1],
                                   self.options['kernel_dims'],
                                   self.options['n_filters_per_block'][block_index], self.options['strides'],
                                   self.options['padding'], self.options['dilation_rate'],
                                   self.options['activation'], self.options['use_bias'],
                                   self.options['batch_normalization'], self.options['drop_out'],
                                   self.variables['feature_maps'][self.options['n_blocks'] - block_index - 1])
                if cfg.VERBOSE:
                    print(' Result is Tensor with shape ', x.shape)
                    print(' -------------------------------------')

        # Decoding
        for block_index in range(7, 9):
            with tf.name_scope('%02d_dec_block' % block_index):
                if cfg.VERBOSE: print(' Decoding Block ', block_index)
                x = block.decoding(x,
                                   self.options, self.options['n_convolutions'][2],
                                   self.options['kernel_dims'],
                                   self.options['n_filters_per_block'][block_index], self.options['strides'],
                                   self.options['padding'], self.options['dilation_rate'],
                                   self.options['activation'], self.options['use_bias'],
                                   self.options['batch_normalization'], self.options['drop_out'],
                                   self.variables['feature_maps'][self.options['n_blocks'] - block_index - 1])
                if cfg.VERBOSE:
                    print(' Result is Tensor with shape ', x.shape)
                    print(' -------------------------------------')

        # Add final 1x1 convolutional layer to compute logits
        with tf.name_scope('10_last_layer'):
            self.outputs['probabilities'] = layer.last(x, np.ones(self.options['rank'], dtype=np.int32), self.options['out_channels'], self.options['strides'],
                                                self.options['padding'], self.options['dilation_rate'],
                                                self._select_final_activation(), False, self.options['use_cross_hair'], do_summary=True)
            if cfg.VERBOSE:
                print(' Probabilities has shape ', self.outputs['probabilities'].shape)
                print(' -------------------------------------')

        return tf.keras.Model(inputs=self.inputs['x'], outputs=self.outputs['probabilities'])


class ResNet(SegBasisNet):
    '''!
    U-Net TODO: add reference

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
                 n_filters=[64, 128, 256, 512, 1024], kernel_dims=3, n_convolutions=[2, 3, 2], drop_out=[False, 0.8],
                 regularize=[True, 'L2', 0.00001], do_batch_normalization=False, do_bias=True,
                 activation='relu', upscale='TRANS_CONV', downscale='MAX_POOL', res_connect=True, skip_connect=False,
                 cross_hair=False):
            super(ResNet, self).__init__(loss, is_training, do_finetune, model_path,
                 n_filters, kernel_dims, n_convolutions, drop_out,
                 regularize, do_batch_normalization, do_bias,
                 activation, upscale, downscale, res_connect, skip_connect, cross_hair)

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

        if cfg.VERBOSE:
            self._print_init()

        x = self.inputs['x']

        # Encoding
        for block_index in range(0, 2):
            with tf.name_scope('%02d_enc_block' % (block_index)):
                if cfg.VERBOSE: print(' Encoding Block ', block_index)
                x = block.encoding(x, self.options, self.variables, self.options['n_convolutions'][0],
                                   self.options['kernel_dims_per_block'][block_index],
                                   self.options['n_filters_per_block'][block_index], [1, 1],
                                   self.options['padding_per_block'][block_index], self.options['dilation_rate'],
                                   self.options['activation_per_block'][block_index], self.options['use_bias'],
                                   self.options['batch_normalization_per_block'][block_index], self.options['drop_out'])
                # self.variables['feature_maps'].append(x) is performed in the encoding block
                if cfg.VERBOSE:
                    print(' Result is Tensor with shape ', x.shape)
                    print(' -------------------------------------')

        # Encoding
        for block_index in range(2, 4):
            with tf.name_scope('%02d_enc_block' % (block_index)):
                if cfg.VERBOSE: print(' Encoding Block ', block_index)
                x = block.encoding(x, self.options, self.variables, self.options['n_convolutions'][1],
                                   self.options['kernel_dims_per_block'][block_index],
                                   self.options['n_filters_per_block'][block_index], self.options['dilation_rate'],
                                   self.options['padding_per_block'][block_index], self.options['dilation_rate'],
                                   self.options['activation_per_block'][block_index], self.options['use_bias'],
                                   self.options['batch_normalization_per_block'][block_index], self.options['drop_out'])
                # self.variables['feature_maps'].append(x) is performed in the encoding block
                if cfg.VERBOSE:
                    print(' Result is Tensor with shape ', x.shape)
                    print(' -------------------------------------')

        # Bottom
        block_index = 4
        with tf.name_scope('%02d_bot_block' % block_index):
            if cfg.VERBOSE: print(' Bottom Block ', block_index)
            x = block.basic(x, self.options, self.options['n_convolutions'][1],
                            self.options['kernel_dims_per_block'][block_index],
                            self.options['n_filters_per_block'][block_index], self.options['dilation_rate'],
                            self.options['padding_per_block'][block_index], self.options['dilation_rate'],
                            self.options['activation_per_block'][block_index], self.options['use_bias'],
                            self.options['batch_normalization_per_block'][block_index], self.options['drop_out'])

            if cfg.VERBOSE:
                print(' Result is Tensor with shape ', x.shape)
                print(' -------------------------------------')

        # Decoding
        for block_index in range(5, 7):
            with tf.name_scope('%02d_dec_block' % block_index):
                if cfg.VERBOSE: print(' Decoding Block ', block_index)
                x = block.decoding(x,
                                 self.options, self.options['n_convolutions'][1],
                                 self.options['kernel_dims_per_block'][block_index],
                                 self.options['n_filters_per_block'][block_index], self.options['dilation_rate'],
                                 self.options['padding_per_block'][block_index], self.options['dilation_rate'],
                                 self.options['activation_per_block'][block_index], self.options['use_bias'],
                                 self.options['batch_normalization_per_block'][block_index], self.options['drop_out'])

                if cfg.VERBOSE:
                    print(' Result is Tensor with shape ', x.shape)
                    print(' -------------------------------------')

        # Decoding
        for block_index in range(7, 9):
            with tf.name_scope('%02d_dec_block' % block_index):
                if cfg.VERBOSE: print(' Decoding Block ', block_index)
                x = block.decoding(x,
                                   self.options, self.options['n_convolutions'][2],
                                   self.options['kernel_dims_per_block'][block_index],
                                   self.options['n_filters_per_block'][block_index], self.options['dilation_rate'],
                                   self.options['padding_per_block'][block_index], self.options['dilation_rate'],
                                   self.options['activation_per_block'][block_index], self.options['use_bias'],
                                   self.options['batch_normalization_per_block'][block_index], self.options['drop_out'])
                if cfg.VERBOSE:
                    print(' Result is Tensor with shape ', x.shape)
                    print(' -------------------------------------')

        # Add final 1x1 convolutional layer to compute logits
        with tf.name_scope('10_last_layer'):
            self.outputs['probabilities'] = layer.last(x, np.ones(self.options['rank'], dtype=np.int32),
                                                       self.options['out_channels'], self.options['strides'],
                                                       self.options['padding'], self.options['dilation_rate'],
                                                       self._select_final_activation(), False,
                                                       self.options['use_cross_hair'], do_summary=True)
            if cfg.VERBOSE:
                print(' Probabilities has shape ', self.outputs['probabilities'].shape)
                print(' -------------------------------------')

        return tf.keras.Model(inputs=self.inputs['x'], outputs=self.outputs['probabilities'])