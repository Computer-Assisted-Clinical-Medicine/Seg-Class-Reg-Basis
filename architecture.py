"""Collection of architectures that can be used for segmentation
this can be uncommented to customize the model
class Model(tf.keras.models.Model):
    '''
    This can be used to customize the training loop or the other training steps.
    '''
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
"""
import logging

import tensorflow as tf
from SegmentationArchitectures import deeplab, densenets, unets
from tensorflow.keras.models import Model

from . import config as cfg
from .segbasisnet import SegBasisNet

# configure logger
logger = logging.getLogger(__name__)

# super-delegation is not useless, it sets default values.
# Public methods are inherited.
# pylint: disable=useless-super-delegation,too-few-public-methods


class UNet(SegBasisNet):
    """
    Implements the UNet.
    inspired by https://arxiv.org/abs/1505.04597
    """

    def __init__(
        self,
        loss_name,
        is_training=True,
        do_finetune=False,
        model_path=None,
        n_filters=(8, 16, 32, 64, 128),
        kernel_dims=3,
        n_convolutions=(2, 2, 3, 3, 3),
        attention=False,
        encoder_attention=None,
        drop_out=(False, 0.2),
        regularize=(True, "L2", 0.00001),
        do_batch_normalization=False,
        do_instance_normalization=False,
        do_bias=True,
        activation="relu",
        upscale="TRANS_CONV",
        downscale="MAX_POOL",
        res_connect=False,
        res_connect_type="skip_first",
        ratio=1,
        skip_connect=True,
        cross_hair=False,
        **kwargs,
    ):

        if do_bias:
            if do_batch_normalization or do_instance_normalization:
                print("use no bias with this network, bias set to False")
                do_bias = False

        super().__init__(
            # standard parameters
            loss_name=loss_name,
            is_training=is_training,
            do_finetune=do_finetune,
            model_path=model_path,
            drop_out=drop_out,
            regularize=regularize,
            activation=activation,
            # model parameters
            n_filters=n_filters,
            kernel_dims=kernel_dims,
            n_convolutions=n_convolutions,
            attention=attention,
            encoder_attention=encoder_attention,
            batch_normalization=do_batch_normalization,
            instance_normalization=do_instance_normalization,
            use_bias=do_bias,
            res_connect=res_connect,
            res_connect_type=res_connect_type,
            skip_connect=skip_connect,
            upscale=upscale,
            downscale=downscale,
            ratio=ratio,
            cross_hair=cross_hair,
            **kwargs,
        )

        # derive some further parameters
        if self.options["batch_normalization"] and self.options["use_bias"]:
            logger.warning("Caution: do not use bias AND batch normalization!")
        if self.options["upscale"] == "UNPOOL_MAX_IND":
            self.variables["unpool_params"] = []
            self.options["downscale"] = "MAX_POOL_ARGMAX"
            if downscale != "MAX_POOL_ARGMAX":
                logger.warning("Caution: changed downscale to MAX_POOL_ARGMAX!")
        else:
            if downscale == "MAX_POOL_ARGMAX":
                raise ValueError("MAX_POOL_ARGMAX has to be used with UNPOOL_MAX_IND!")
        if self.options["rank"] == 2 and self.options["cross_hair"]:
            logger.warning("Caution: cross_hair is ignored for 2D input!")
        if self.options["skip_connect"]:
            self.variables["feature_maps"] = []

        # each block is followed by one pooling operation (except the bottleneck)
        self.divisible_by = 2 ** (len(n_filters) - 1)

    @staticmethod
    def get_name():
        return "UNet"

    def _build_model(self) -> Model:
        """Builds UNet"""

        return unets.unet(
            input_tensor=self.inputs["x"],
            out_channels=self.options["out_channels"],
            loss=self.options["loss_name"],
            n_filter=self.options["n_filters"],
            n_convolutions=self.options["n_convolutions"],
            attention=self.options["attention"],
            encoder_attention=self.options["encoder_attention"],
            kernel_dims=self.options["kernel_dims"],
            stride=1,
            batch_normalization=self.options["batch_normalization"],
            instance_normalization=self.options["instance_normalization"],
            use_bias=self.options["use_bias"],
            drop_out=self.options["drop_out"],
            upscale=self.options["upscale"],
            downscale=self.options["downscale"],
            regularize=self.options["regularize"],
            padding="SAME",
            activation=self.options["activation"],
            name="Unet",
            ratio=self.options["ratio"],
            dilation_rate=1,
            cross_hair=self.options["cross_hair"],
            res_connect=self.options["res_connect"],
            res_connect_type=self.options["res_connect_type"],
            skip_connect=self.options["skip_connect"],
        )


class DenseTiramisu(SegBasisNet):
    """
    Implements the One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation.
    inspired by https://github.com/SimJeg/FC-DenseNet

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

    def __init__(
        self,
        loss_name,
        is_training=True,
        kernel_dims=3,
        growth_rate=16,
        layers_per_block=(4, 5, 7, 10, 12),
        bottleneck_layers=15,
        drop_out=(True, 0.2),
        regularize=(True, "L2", 0.001),
        do_batch_normalization=True,
        do_bias=False,
        activation="relu",
        **kwargs,
    ):
        if do_bias:
            print("use no bias with this network, bias set to False")
            do_bias = False
        if not do_batch_normalization:
            print("always uses batch norm, set to True")
            do_batch_normalization = True

        super().__init__(
            loss_name=loss_name,
            is_training=is_training,
            kernel_dims=kernel_dims,
            growth_rate=growth_rate,
            layers_per_block=layers_per_block,
            bottleneck_layers=bottleneck_layers,
            drop_out=drop_out,
            regularize=regularize,
            do_bias=False,
            do_batch_normalization=True,
            activation=activation,
            **kwargs,
        )

        # each block is followed by one pooling operation
        self.divisible_by = 2 ** len(layers_per_block)

    @staticmethod
    def get_name():
        return "DenseTiramisu"

    def _build_model(self) -> Model:
        """Builds DenseTiramisu"""

        return densenets.DenseTiramisu(
            input_tensor=self.inputs["x"],
            out_channels=self.options["out_channels"],
            loss=self.options["loss_name"],
            is_training=self.options["is_training"],
            kernel_dims=self.options["kernel_dims"],
            growth_rate=self.options["growth_rate"],
            layers_per_block=self.options["layers_per_block"],
            bottleneck_layers=self.options["bottleneck_layers"],
            drop_out=self.options["drop_out"],
            activation=self.options["activation"],
        )


class DeepLabv3plus(SegBasisNet):
    """
    Implements DeepLabv3plus.
    inspired by https://github.com/srihari-humbarwadi/DeepLabV3_Plus-Tensorflow2.0
    and https://github.com/bonlime/keras-deeplab-v3-plus

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

    def __init__(
        self,
        loss_name,
        is_training=True,
        kernel_dims=3,
        drop_out=(True, 0.2),
        regularize=(True, "L2", 0.001),
        backbone="resnet50",
        aspp_rates=(6, 12, 18),
        do_batch_normalization=True,
        do_bias=False,
        activation="relu",
        **kwargs,
    ):
        if do_bias:
            print("use no bias with this network, bias set to False")
            do_bias = False
        if not do_batch_normalization:
            print("always uses batch norm, set to True")
            do_batch_normalization = True

        super().__init__(
            loss_name=loss_name,
            is_training=is_training,
            kernel_dims=kernel_dims,
            drop_out=drop_out,
            regularize=regularize,
            backbone=backbone,
            aspp_rates=aspp_rates,
            do_bias=False,
            do_batch_normalization=True,
            activation=activation,
            **kwargs,
        )

        # last layer should be 16 times smaller than the input
        self.divisible_by = 16

    @staticmethod
    def get_name():
        return "DeepLabv3plus"

    def set_up_inputs(self):
        """setup the inputs. Inputs are taken from the config file."""
        self.inputs["x"] = tf.keras.Input(
            shape=cfg.train_input_shape,
            batch_size=None,
            dtype=cfg.dtype,
            name="input",
        )
        self.options["out_channels"] = cfg.num_classes_seg

    def _build_model(self) -> Model:
        """Builds DeepLabv3plus"""

        return deeplab.DeepLabv3plus(
            input_tensor=self.inputs["x"],
            out_channels=self.options["out_channels"],
            loss=self.options["loss_name"],
            is_training=self.options["is_training"],
            kernel_dims=self.options["kernel_dims"],
            drop_out=self.options["drop_out"],
            regularize=self.options["regularize"],
            backbone=self.options["backbone"],
            aspp_rates=self.options["aspp_rates"],
            activation=self.options["activation"],
            debug=self.options.get("debug", False),
        )
