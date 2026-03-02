"""Z-axis subsampling augmentations for super-resolution training."""
from __future__ import print_function
import numpy as np

from .augment import Augment


__all__ = ['SubsampleZ', 'CubicSubsampleZ']


class SubsampleZ(Augment):
    """Subsample data in Z dimension.

    Expands the required spec in Z during prepare(), then subsamples
    during __call__(). Input is average-pooled, labels use nearest neighbor.

    Args:
        factor: Subsampling factor in Z dimension.
        imgs: Keys to treat as images (average pooling).
        segs: Keys to treat as segmentation (nearest neighbor).
    """
    def __init__(self, factor, imgs=['input'], segs=None):
        self.factor = factor
        self.imgs = imgs
        self.segs = segs if segs is not None else []

    def prepare(self, spec, imgs=[], segs=[], **kwargs):
        # Update imgs/segs from kwargs if provided
        if imgs:
            self.imgs = imgs
        if segs:
            self.segs = segs

        # Expand spec in Z dimension
        spec = dict(spec)
        for k, dims in spec.items():
            z, y, x = dims[-3:]
            spec[k] = tuple(dims[:-3]) + (z * self.factor, y, x)
        return spec

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        for k, v in sample.items():
            if k in self.imgs or k == 'input':
                sample[k] = self._subsample_avg(v)
            elif k in self.segs or k.endswith('_mask'):
                sample[k] = self._subsample_nearest(v)
            else:
                # Default: use nearest for unknown keys (safer for labels)
                sample[k] = self._subsample_nearest(v)
        return Augment.sort(sample)

    def _subsample_avg(self, data):
        """Subsample using average pooling."""
        # data shape: (C, Z, Y, X) or (Z, Y, X)
        if data.ndim == 3:
            z, y, x = data.shape
            new_z = z // self.factor
            data = data.reshape(new_z, self.factor, y, x).mean(axis=1)
        else:
            c, z, y, x = data.shape
            new_z = z // self.factor
            data = data.reshape(c, new_z, self.factor, y, x).mean(axis=2)
        return data

    def _subsample_nearest(self, data):
        """Subsample by taking the middle slice of each block."""
        offset = self.factor // 2
        if data.ndim == 3:
            return data[offset::self.factor, :, :]
        else:
            return data[:, offset::self.factor, :, :]

    def __repr__(self):
        return f'{self.__class__.__name__}(factor={self.factor})'


class CubicSubsampleZ(Augment):
    """Expand to cubic patch for isotropic augmentation, then subsample input.

    This augmentation enables FlipRotateIsotropic on data that will be
    subsampled in Z. It works by:
    1. prepare(): Expand aniso spec to a cube (max dimension, rounded to
       factor multiple) so upstream iso augments operate on cubic data.
    2. __call__(): Subsample input in Z and crop to aniso target size.
       Labels/masks are center-cropped to iso target size (no subsampling).

    The spec passed to prepare() should have aniso Z dimensions. This class
    computes the iso target as (z_aniso * factor, y, x) and the cubic size
    from the maximum of all iso dimensions.

    Args:
        factor: Subsampling factor in Z dimension.
        imgs: Keys to treat as images (subsample + crop to aniso).
    """
    def __init__(self, factor, imgs=['input']):
        self.factor = factor
        self.imgs = imgs
        self.target_spec = None
        self.cubic_size = None
        self.iso_target = None

    def prepare(self, spec, imgs=[], **kwargs):
        if imgs:
            self.imgs = imgs

        # Store aniso target spec for cropping input after subsampling
        self.target_spec = dict(spec)

        # Compute iso target (z_aniso * factor, y, x) for cropping labels
        self.iso_target = {}
        for k, dims in spec.items():
            z, y, x = dims[-3:]
            self.iso_target[k] = (z * self.factor, y, x)

        # Compute cubic size from the max iso dimension
        cubic_spec = {}
        for k, dims in spec.items():
            z, y, x = dims[-3:]
            iso_z = z * self.factor
            max_dim = max(iso_z, y, x)
            # Round up to multiple of factor for clean subsampling
            self.cubic_size = ((max_dim + self.factor - 1) // self.factor) * self.factor
            cubic_spec[k] = tuple(dims[:-3]) + (self.cubic_size,) * 3

        return cubic_spec

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        result = {}

        for k, v in sample.items():
            if k in self.imgs:
                # Input: subsample in Z, then crop to aniso target
                v_sub = self._subsample_avg(v)
                result[k] = self._center_crop(v_sub, self.target_spec[k][-3:])
            else:
                # Labels/masks: crop to iso target (no subsampling)
                base_key = k.replace('_mask', '') if k.endswith('_mask') else k
                iso_target = self.iso_target.get(k) or self.iso_target.get(base_key)
                if iso_target is None:
                    result[k] = v
                else:
                    result[k] = self._center_crop(v, iso_target)

        return Augment.sort(result)

    def _subsample_avg(self, data):
        """Subsample using average pooling in Z."""
        if data.ndim == 3:
            z, y, x = data.shape
            new_z = z // self.factor
            return data.reshape(new_z, self.factor, y, x).mean(axis=1)
        else:
            c, z, y, x = data.shape
            new_z = z // self.factor
            return data.reshape(c, new_z, self.factor, y, x).mean(axis=2)

    @staticmethod
    def _center_crop(data, target_shape):
        """Center crop spatial dimensions to target shape."""
        tz, ty, tx = target_shape
        if data.ndim == 3:
            z, y, x = data.shape
            sz, sy, sx = (z - tz) // 2, (y - ty) // 2, (x - tx) // 2
            return data[sz:sz+tz, sy:sy+ty, sx:sx+tx]
        else:
            _, z, y, x = data.shape
            sz, sy, sx = (z - tz) // 2, (y - ty) // 2, (x - tx) // 2
            return data[:, sz:sz+tz, sy:sy+ty, sx:sx+tx]

    def __repr__(self):
        return f'{self.__class__.__name__}(factor={self.factor})'
