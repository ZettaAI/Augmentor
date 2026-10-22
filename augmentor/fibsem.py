"""Noise of a fast FIB-SEM acquisition, as measured on a real one.

A short pixel dwell leaves three marks on a FIB-SEM image that a slow,
heavily dosed one (e.g. the Janelia hemibrain) lacks:

1. Shot noise whose variance depends on intensity. FIB-SEM is shown with
   inverted contrast -- heavy-metal stain gives more electrons but a darker
   grey -- so the variance grows toward the dark end:
   ``var(I) = noise_var + dark_slope * (1 - I)`` (``invert=True``).
2. Noise correlated along the fast scan axis: at a high pixel rate the
   detector's bandwidth blurs each scan line, so the noise is white noise
   filtered by a Gaussian of width ``sigma_x`` along that axis.
3. Per-slice gain jitter: every milled slice is imaged with a slightly
   different beam current or detector gain, a multiplicative change of the
   inverted signal, uniform across the slice and independent between slices.

Fitted to the winding fly larva volume P667 (Crick) on clean training cubes;
see winding_fly_larva/analysis/fibsem_calibrate.py:

======  =========  ==========  =======  ==========  =====================
voxel   noise_var  dark_slope  sigma_x  slice_gain  fitted on
======  =========  ==========  =======  ==========  =====================
16 nm   9.1e-5     4.2e-4      0.53     0.029       hemibrain pb2 (default)
8 nm    2.5e-4     1.15e-3     0.71     0.024       Chklovskii wasp r00
======  =========  ==========  =======  ==========  =====================

All parameters are per voxel at the resolution the noise is drawn on. The
16 nm preset is not the 8 nm one averaged down: P667's noise is
anti-correlated between neighbouring slices and averages away faster than
this model does, so the 16 nm values were fitted on P667's own 16 nm image.
``oversample=2`` draws the noise on a grid twice as fine and averages it
down; it reproduces degrading at 8 nm and then averaging.
"""
from __future__ import print_function
from collections import OrderedDict

import numpy as np
from scipy.ndimage import convolve1d

from .augment import Augment


__all__ = ['FIBSEMNoise', 'fibsem_noise', 'FIBSEM_LEVELS']


# Named severities on top of the 16 nm preset. "winding" is P667 itself; the
# others multiply its noise variance by `scale` and widen the detector kernel
# and the slice gain, with `scale` chosen so each level has the same effective
# SNR as the ImageDegradation level of the same name, on Chklovskii r00 at
# 16 nm (winding_fly_larva/analysis/degrade_levels.py).
FIBSEM_LEVELS = OrderedDict([
    ('winding',   dict(scale=1.0,   sigma_x=0.53, slice_gain=0.029)),
    ('mild',      dict(scale=23.0,  sigma_x=0.6,  slice_gain=0.04)),
    ('moderate',  dict(scale=48.0,  sigma_x=0.7,  slice_gain=0.05)),
    ('severe',    dict(scale=97.0,  sigma_x=0.8,  slice_gain=0.06)),
    ('harsh',     dict(scale=133.0, sigma_x=0.9,  slice_gain=0.07)),
    ('brutal',    dict(scale=210.0, sigma_x=1.0,  slice_gain=0.08)),
    ('ultra',     dict(scale=235.0, sigma_x=1.1,  slice_gain=0.09)),
    ('nightmare', dict(scale=320.0, sigma_x=1.2,  slice_gain=0.10)),
])


def _uniform(r):
    lo, hi = (r, r) if np.isscalar(r) else r
    return float(np.random.uniform(lo, hi))


def _loguniform(r):
    lo, hi = (r, r) if np.isscalar(r) else r
    return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))


def _gaussian_kernel(sigma, truncate=4.0):
    """Unit-energy Gaussian FIR, so filtered white noise keeps its variance."""
    if sigma <= 0:
        return np.ones(1, dtype=np.float32)
    radius = max(1, int(np.ceil(truncate * sigma)))
    x = np.arange(-radius, radius + 1)
    h = np.exp(-0.5 * (x / sigma) ** 2)
    return (h / np.sqrt((h ** 2).sum())).astype(np.float32)


def _block_mean(x, k):
    """Average k x k x k blocks of the last three axes."""
    if k == 1:
        return x
    z, y, w = x.shape[-3:]
    shape = x.shape[:-3] + (z // k, k, y // k, k, w // k, k)
    return x.reshape(shape).mean(axis=(-5, -3, -1))


def fibsem_noise(img, noise_var, dark_slope, sigma_x, slice_gain,
                 invert=True, oversample=1, fast_axis=-1, white=1.0):
    """Degrade ``img`` (..., z, y, x), float in [0, 1], in place.

    Args:
        noise_var: variance at the bright end (I = 1 when ``invert``).
        dark_slope: added variance per unit of darkness, (1 - I).
        sigma_x: Gaussian width, in voxels of the fine grid, along the fast
            scan axis.
        slice_gain: standard deviation of the per-slice gain.
        invert: variance and gain act on the inverted signal (white - I).
        oversample: draw on a grid this many times finer per axis, then
            average down.
        fast_axis: axis of ``img`` (one of the last three) that is scanned.
        white: the intensity of zero signal when ``invert``.
    """
    k = int(oversample)
    signal = (white - img) if invert else img

    # 3. per-slice gain: one draw per fine slice, averaged per coarse slice.
    if slice_gain > 0:
        nz = img.shape[-3]
        g = 1 + slice_gain * np.random.randn(nz * k).astype(np.float32)
        g = g.reshape(nz, k).mean(axis=1)
        g = g.reshape((nz, 1, 1))
        if invert:
            img[...] = white - signal * g
        else:
            img[...] = signal * g
        signal = (white - img) if invert else img

    # 1 + 2. intensity-dependent noise, correlated along the fast axis.
    var = noise_var + dark_slope * np.clip(signal, 0, None)
    if k > 1:
        for axis in (-3, -2, -1):
            var = np.repeat(var, k, axis=axis)
    noise = np.random.randn(*var.shape).astype(np.float32)
    noise = convolve1d(noise, _gaussian_kernel(sigma_x), axis=fast_axis,
                       mode='wrap')
    noise *= np.sqrt(np.clip(var, 0, None)).astype(np.float32)
    img += _block_mean(noise, k)
    np.clip(img, 0, 1, out=img)
    return img


class FIBSEMNoise(Augment):
    """Fast FIB-SEM acquisition noise (see the module docstring).

    Each range is drawn once per sample; a scalar fixes the value.

    The defaults train far harder than P667: the variance multiplier spans
    the "winding" level to "brutal" (``FIBSEM_LEVELS``), log-uniformly.

    Args:
        noise_var (tuple): variance at the bright end, [0, 1] intensity units.
        dark_slope (tuple): added variance per unit of darkness.
        scale (tuple): log-uniform multiplier on both variance terms.
        sigma_x (tuple): fast-axis correlation width, in voxels of the grid
            the noise is drawn on (the fine grid when oversampling).
        slice_gain (tuple): standard deviation of the per-slice gain.
        invert (bool): inverted display contrast, as FIB-SEM is shown.
        oversample (int): noise drawn on a grid this many times finer.
        fast_axis (int): the scanned axis of each image, one of -3, -2, -1.
            Apply this before any flip or transpose, so it is the
            acquisition's fast axis.
        skip (float): probability of leaving the sample unchanged.
    """
    def __init__(self, noise_var=9.1e-5, dark_slope=4.2e-4, scale=(1.0, 210.0),
                 sigma_x=(0.3, 1.0), slice_gain=(0.0, 0.08), invert=True,
                 oversample=1, fast_axis=-1, skip=0.3):
        assert fast_axis in (-3, -2, -1)
        self.noise_var = noise_var
        self.dark_slope = dark_slope
        self.scale = scale
        self.sigma_x = sigma_x
        self.slice_gain = slice_gain
        self.invert = invert
        self.oversample = int(oversample)
        self.fast_axis = fast_axis
        self.skip = np.clip(skip, 0, 1)
        self.do_aug = False
        self.imgs = []
        self.params = {}

    @classmethod
    def at_level(cls, level, **kwargs):
        """One named severity of ``FIBSEM_LEVELS``, every draw the same."""
        params = dict(FIBSEM_LEVELS[level], skip=0.0)
        params.update(kwargs)
        return cls(**params)

    def prepare(self, spec, imgs=[], **kwargs):
        self.do_aug = np.random.rand() > self.skip
        self.imgs = self._validate(spec, imgs)
        if self.do_aug:
            s = _loguniform(self.scale)
            self.params = dict(
                noise_var=s * _uniform(self.noise_var),
                dark_slope=s * _uniform(self.dark_slope),
                sigma_x=_uniform(self.sigma_x),
                slice_gain=_uniform(self.slice_gain),
            )
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.do_aug:
            for k in self.imgs:
                img = sample[k]
                if img.dtype != np.float32:
                    img = img.astype(np.float32)
                fibsem_noise(img, invert=self.invert,
                             oversample=self.oversample,
                             fast_axis=self.fast_axis, **self.params)
                sample[k] = img
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'noise_var={}, '.format(self.noise_var)
        format_string += 'dark_slope={}, '.format(self.dark_slope)
        format_string += 'scale={}, '.format(self.scale)
        format_string += 'sigma_x={}, '.format(self.sigma_x)
        format_string += 'slice_gain={}, '.format(self.slice_gain)
        format_string += 'invert={}, '.format(self.invert)
        format_string += 'oversample={}, '.format(self.oversample)
        format_string += 'fast_axis={}, '.format(self.fast_axis)
        format_string += 'skip={:.2f}'.format(self.skip)
        format_string += ')'
        return format_string

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        return imgs
