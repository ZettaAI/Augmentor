from __future__ import print_function
import numpy as np

from .augment import Augment, Blend
from .perturb import ContrastCompression
from .section import Section, PartialSection, MixedSection


__all__ = ['ContrastCompression2D', 'ContrastCompression3D',
           'ContrastCompressionMixed', 'PartialContrastCompression2D',
           'MixedContrastCompression2D']


class ContrastCompression3D(Augment):
    """Contrast compression augmentation for 3D volumes.

    Simulates low SNR/contrast regions by compressing the intensity
    distribution toward the mean. Applies the same transformation
    uniformly across the entire 3D volume.

    Args:
        compression_factor (tuple): Range (min, max) to sample compression
            factor from. Values < 1 compress the distribution.
        mean_shift (tuple): Range (min, max) to sample mean shift from.
        use_local_mean (bool): If True, compute mean from input image.
        skip (float): Probability of skipping the augmentation.
    """
    def __init__(self, compression_factor=(0.4, 0.7), mean_shift=(-0.1, 0.1),
                 use_local_mean=True, skip=0.3):
        self.compression_factor = compression_factor
        self.mean_shift = mean_shift
        self.use_local_mean = use_local_mean
        self.skip = np.clip(skip, 0, 1)
        self.do_aug = False
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        # Biased coin toss.
        self.do_aug = np.random.rand() > self.skip
        self.imgs = self._validate(spec, imgs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.do_aug:
            perturb = ContrastCompression(
                self.compression_factor,
                self.mean_shift,
                self.use_local_mean
            )
            for k in self.imgs:
                perturb(sample[k])
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'compression_factor={}, '.format(self.compression_factor)
        format_string += 'mean_shift={}, '.format(self.mean_shift)
        format_string += 'use_local_mean={}, '.format(self.use_local_mean)
        format_string += 'skip={:.2f}'.format(self.skip)
        format_string += ')'
        return format_string

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        return imgs


class ContrastCompression2D(Section):
    """Contrast compression applied per z-slice.

    Each z-slice can receive a different compression transformation,
    simulating spatially varying image quality degradation.
    """
    def __init__(self, compression_factor=(0.4, 0.7), mean_shift=(-0.1, 0.1),
                 use_local_mean=True, prob=1, **kwargs):
        super(ContrastCompression2D, self).__init__(
            ContrastCompression, prob=prob, **kwargs
        )
        self.params = dict(
            compression_factor=compression_factor,
            mean_shift=mean_shift,
            use_local_mean=use_local_mean
        )


class ContrastCompressionMixed(Blend):
    """Half 2D & half 3D contrast compression."""
    def __init__(self, **kwargs):
        augments = [
            ContrastCompression2D(**kwargs),
            ContrastCompression3D(**kwargs)
        ]
        super(ContrastCompressionMixed, self).__init__(augments)


class PartialContrastCompression2D(PartialSection):
    """Partial contrast compression per z-slice.

    Applies contrast compression to random quadrants of each slice,
    simulating localized image quality degradation.
    """
    def __init__(self, compression_factor=(0.4, 0.7), mean_shift=(-0.1, 0.1),
                 use_local_mean=True, **kwargs):
        super(PartialContrastCompression2D, self).__init__(
            ContrastCompression, **kwargs
        )
        self.params = dict(
            compression_factor=compression_factor,
            mean_shift=mean_shift,
            use_local_mean=use_local_mean
        )


class MixedContrastCompression2D(MixedSection):
    """Mixed full/partial contrast compression per z-slice."""
    def __init__(self, compression_factor=(0.4, 0.7), mean_shift=(-0.1, 0.1),
                 use_local_mean=True, **kwargs):
        super(MixedContrastCompression2D, self).__init__(
            ContrastCompression, **kwargs
        )
        self.params = dict(
            compression_factor=compression_factor,
            mean_shift=mean_shift,
            use_local_mean=use_local_mean
        )
