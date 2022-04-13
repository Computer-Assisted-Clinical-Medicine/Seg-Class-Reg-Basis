"""
This module takes the existing base loaders and adapts them to the current task
"""
import logging

import SimpleITK as sitk

from . import config as cfg
from .segapplyloader import ApplyBasisLoader
from .segbasisloader import SegBasisLoader

# configure logger
logger = logging.getLogger(__name__)


class SegLoader(SegBasisLoader):
    """
    Interface the same as SegBasisLoader
    """

    def adapt_to_task(self, data_img: sitk.Image, label_img: sitk.Image):
        """Make sure that labels have type uint-8 and only use labels 0 or 1

        Parameters
        ----------
        data_img : sitk.Image
            The data image
        label_img : sitk.Image
            The label image

        Returns
        -------
        sitk.Image, sitk.Image
            The converted images
        """
        if label_img is not None:
            label_img = sitk.Threshold(
                label_img,
                upper=cfg.num_classes_seg - 1,
                outsideValue=cfg.num_classes_seg - 1,
            )
            # label should be uint-8
            if label_img.GetPixelID() != sitk.sitkUInt8:
                label_img = sitk.Cast(label_img, sitk.sitkUInt8)
        return data_img, label_img


class ApplyLoader(ApplyBasisLoader):
    """
    Interface the same as ApplyBasisLoader
    """

    def adapt_to_task(self, data_img: sitk.Image, label_img: sitk.Image):
        """Make sure that labels have type uint-8 and only use labels 0 or 1

        Parameters
        ----------
        data_img : sitk.Image
            The data image
        label_img : sitk.Image
            The label image

        Returns
        -------
        sitk.Image, sitk.Image
            The converted images
        """
        if label_img is not None:
            label_img = sitk.Threshold(
                label_img,
                upper=cfg.num_classes_seg - 1,
                outsideValue=cfg.num_classes_seg - 1,
            )
            # label should be uint-8
            if label_img.GetPixelID() != sitk.sitkUInt8:
                label_img = sitk.Cast(label_img, sitk.sitkUInt8)
        return data_img, label_img
