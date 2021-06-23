'''Collection of architecures that can be used for segmentation
'''
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from SegmentationArchitectures import densenets
from SegmentationArchitectures import deeplab
from . import config as cfg
from .NetworkBasis import block, layer
from .segbasisnet import SegBasisNet


#configure logger
logger = logging.getLogger(__name__)

# this can be uncommented to customize the model
# class Model(tf.keras.models.Model):
#     '''
#     This can be used to customize the training loop or the other training steps.
#     '''
#     def __init__(self, **kwargs):
#         super(Model, self).__init__(**kwargs)


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
                 n_filters=(8, 16, 32, 64, 128), kernel_dims=3, n_convolutions=(2, 3, 2), drop_out=(False, 0.2),
                 regularize=(True, 'L2', 0.00001), do_batch_normalization=False, do_bias=True,
                 activation='relu', upscale='TRANS_CONV', downscale='MAX_POOL', res_connect=False, skip_connect=True,
                 cross_hair=False, **kwargs):
        super(UNet, self).__init__(loss, is_training, do_finetune, model_path,
                 n_filters, kernel_dims, n_convolutions, drop_out,
                 regularize, do_batch_normalization, do_bias,
                 activation, upscale, downscale, res_connect, skip_connect, cross_hair, **kwargs)

    @staticmethod
    def get_name():
        return 'UNet'

    def _build_model(self) -> Model:
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
                                   self.options['batch_normalization'], self.options['drop_out'],
                                   name=f'UNet{self.options["rank"]}D-encode{block_index}')
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
                                   self.options['batch_normalization'], self.options['drop_out'],
                                   name=f'UNet{self.options["rank"]}D-encode{block_index}')
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
                            self.options['batch_normalization'], self.options['drop_out'],
                            name=f'UNet{self.options["rank"]}D-bottleneck')
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
                                   self.variables['feature_maps'][self.options['n_blocks'] - block_index - 1],
                                   name=f'UNet{self.options["rank"]}D-decode{block_index}')
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
                                   self.variables['feature_maps'][self.options['n_blocks'] - block_index - 1],
                                   name=f'UNet{self.options["rank"]}D-decode{block_index}')
                logger.debug(' Result is Tensor with shape %s', x.shape)

        # Add final 1x1 convolutional layer to compute logits
        with tf.name_scope('9_last_layer' + str(cfg.num_classes_seg)):
            self.outputs['probabilities'] = layer.last(
                x, self.outputs, np.ones(self.options['rank'], dtype=np.int32),
                self.options['out_channels'], self.options['strides'],
                self.options['padding'], self.options['dilation_rate'],
                self._select_final_activation(), False,
                self.options['regularizer'], self.options['use_cross_hair'],
                do_summary=True, name=f'UNet{self.options["rank"]}D/last/Conn{self.options["rank"]}D'
            )
            logger.debug(' Probabilities have shape %s', self.outputs['probabilities'].shape)

        return Model(inputs=self.inputs['x'], outputs=self.outputs['probabilities'])


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
                 n_filters=(8, 16, 32, 64, 128), kernel_dims=5, n_convolutions=(1, 2, 3, 2, 1), drop_out=(True, 0.2),
                 regularize=(True, 'L2', 0.00001), do_batch_normalization=True, do_bias=False,
                 activation='leaky_relu', upscale='TRANS_CONV', downscale='STRIDE', res_connect=True, skip_connect=True,
                 cross_hair=False, **kwargs):
        super(VNet, self).__init__(loss, is_training, do_finetune, model_path,
                 n_filters, kernel_dims, n_convolutions, drop_out,
                 regularize, do_batch_normalization, do_bias,
                 activation, upscale, downscale, res_connect, skip_connect, cross_hair, **kwargs)

    @staticmethod
    def get_name():
        return 'VNet'

    def _build_model(self) -> Model:
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

        return Model(inputs=self.inputs['x'], outputs=self.outputs['probabilities'])

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

        # each block is followed by one pooling operation
        self.divisible_by = 2**len(layers_per_block)

    @staticmethod
    def get_name():
        return 'DenseTiramisu'

    def _build_model(self) -> Model:
        '''!
        Builds DenseTiramisu

        '''
        return densenets.DenseTiramisu(
            input_tensor=self.inputs["x"],
            out_channels=self.options["out_channels"],
            loss=self.options["loss"],
            is_training=self.options['is_training'],
            kernel_dims=self.options["kernel_dims"],
            growth_rate=self.options["growth_rate"],
            layers_per_block=self.options["layers_per_block"],
            bottleneck_layers=self.options["bottleneck_layers"],
            drop_out=self.options["drop_out"],
            activation=self.options["activation"]
        )


class DeepLabv3plus(SegBasisNet):
    '''
    Implements DeepLabv3plus.
    inspired by https://github.com/srihari-humbarwadi/DeepLabV3_Plus-Tensorflow2.0
    and https://github.com/bonlime/keras-deeplab-v3-plus
    '''

    def __init__(self, loss, is_training=True, kernel_dims=3, drop_out=(True, 0.2),
        regularize=(True, 'L2', 0.001), backbone='resnet50', aspp_rates=(6,12,18), do_batch_normalization=True,
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

        super().__init__(loss, is_training, kernel_dims=kernel_dims, drop_out=drop_out,
            regularize=regularize, backbone=backbone, aspp_rates=aspp_rates, do_bias=False, do_batch_normalization=True,
            activation=activation, **kwargs)

    @staticmethod
    def get_name():
        return 'DeepLabv3plus'

    def _build_model(self) -> Model:
        '''!
        Builds DeepLabv3plus

        '''
        return deeplab.DeepLabv3plus(
            input_tensor=self.inputs["x"],
            out_channels=self.options["out_channels"],
            loss=self.options["loss"],
            is_training=self.options['is_training'],
            kernel_dims=self.options["kernel_dims"],
            drop_out=self.options["drop_out"],
            regularize=self.options["regularize"],
            backbone=self.options["backbone"],
            aspp_rates=self.options["aspp_rates"],
            activation=self.options["activation"],
            debug=self.options["debug"]
        )
