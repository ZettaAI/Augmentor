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
    """Expand to cubic patch for isotropic augmentation, then subsample and crop.

    This augmentation enables FlipRotateIsotropic on data that will be
    subsampled in Z. It works by:
    1. prepare(): Expand spec to a cube (max dimension, rounded to factor multiple)
    2. __call__(): Subsample input in Z, crop labels to iso size, create aniso labels

    For super-resolution training, this produces:
    - 'input': subsampled and cropped (aniso resolution for network input)
    - '<key>': cropped only (iso resolution for iso loss)
    - '<key>_aniso': subsampled and cropped (aniso resolution for aniso loss)

    Args:
        factor: Subsampling factor in Z dimension.
        imgs: Keys to treat as images (subsample input).
        segs: Keys to treat as segmentation labels (keep iso + create aniso).
        dual_output: If True, create both iso and aniso versions of labels.
    """
    def __init__(self, factor, imgs=['input'], segs=None, dual_output=True):
        self.factor = factor
        self.imgs = imgs
        self.segs = segs if segs is not None else []
        self.dual_output = dual_output
        self.target_spec = None
        self.cubic_size = None
        self.iso_target = None  # Target size at iso resolution (before subsample)

    def prepare(self, spec, imgs=[], segs=[], **kwargs):
        # Update imgs/segs from kwargs if provided
        if imgs:
            self.imgs = imgs
        if segs:
            self.segs = segs

        # Store target spec for cropping later (this is the aniso target)
        self.target_spec = dict(spec)

        # Compute iso target (before subsampling)
        self.iso_target = {}
        for k, dims in spec.items():
            z, y, x = dims[-3:]
            self.iso_target[k] = (z * self.factor, y, x)

        # Compute cubic size
        cubic_spec = {}
        for k, dims in spec.items():
            z, y, x = dims[-3:]
            iso_z = z * self.factor
            max_dim = max(iso_z, y, x)
            # Round up to multiple of factor for clean subsampling
            self.cubic_size = ((max_dim + self.factor - 1) // self.factor) * self.factor
            cubic_spec[k] = tuple(dims[:-3]) + (self.cubic_size, self.cubic_size, self.cubic_size)

        return cubic_spec

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        result = {}

        for k, v in sample.items():
            if k in self.imgs or k == 'input':
                # Input: subsample and crop to aniso size
                v_sub = self._subsample_avg(v)
                v_crop = self._center_crop(v_sub, self.target_spec[k][-3:])
                result[k] = v_crop
            else:
                # Labels: crop to iso size, optionally create aniso version
                iso_target = self.iso_target.get(k, self.iso_target.get(k.replace('_mask', '')))
                if iso_target is None:
                    # Unknown key, just pass through
                    result[k] = v
                    continue

                # Crop to iso size (no subsampling)
                v_iso = self._center_crop(v, iso_target)
                result[k] = v_iso

                # Create aniso version if dual_output enabled
                if self.dual_output:
                    v_aniso = self._subsample_nearest(v_iso)
                    aniso_key = k + '_aniso' if not k.endswith('_mask') else k.replace('_mask', '_mask_aniso')
                    result[aniso_key] = v_aniso

        # Add is_isotropic flag
        result['is_isotropic'] = np.array([1], dtype=np.float32)

        return Augment.sort(result)

    def _subsample_avg(self, data):
        """Subsample using average pooling."""
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

    def _center_crop(self, data, target_shape):
        """Center crop data to target shape."""
        tz, ty, tx = target_shape

        if data.ndim == 3:
            z, y, x = data.shape
            sz = (z - tz) // 2
            sy = (y - ty) // 2
            sx = (x - tx) // 2
            return data[sz:sz+tz, sy:sy+ty, sx:sx+tx]
        else:
            c, z, y, x = data.shape
            sz = (z - tz) // 2
            sy = (y - ty) // 2
            sx = (x - tx) // 2
            return data[:, sz:sz+tz, sy:sy+ty, sx:sx+tx]

    def __repr__(self):
        return f'{self.__class__.__name__}(factor={self.factor}, dual_output={self.dual_output})'
