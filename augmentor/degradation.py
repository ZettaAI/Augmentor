from __future__ import print_function
from collections import OrderedDict

import numpy as np

from .augment import Augment, Compose, Blend
from .perturb import PoissonNoise, GaussianNoise, ContrastCompression
from .section import Section, PartialSection, MixedSection
from .contrast import ContrastCompression3D, ContrastCompression2D


__all__ = [
    'DEGRADATION_LEVELS',
    'PoissonNoise3D', 'PoissonNoise2D', 'PoissonNoiseMixed',
    'GaussianNoise3D', 'GaussianNoise2D', 'GaussianNoiseMixed',
    'ImageDegradation', 'ImageDegradation3D', 'ImageDegradation2D',
]


# Named severities, from the harsh-degradation study in
# hammerschmith_mouse_mec/notebooks/260116_augmentation_analysis_part6_harsh_degradation.
# The study's advice: train on mild to harsh, brutal to ultra 10-20 % of the
# time, nightmare never.
DEGRADATION_LEVELS = OrderedDict([
    ('mild',      dict(compression_factor=0.7,  peak_electrons=200, variance=0.005)),
    ('moderate',  dict(compression_factor=0.5,  peak_electrons=100, variance=0.01)),
    ('severe',    dict(compression_factor=0.4,  peak_electrons=50,  variance=0.02)),
    ('harsh',     dict(compression_factor=0.35, peak_electrons=30,  variance=0.025)),
    ('brutal',    dict(compression_factor=0.25, peak_electrons=20,  variance=0.04)),
    ('ultra',     dict(compression_factor=0.2,  peak_electrons=15,  variance=0.04)),
    ('nightmare', dict(compression_factor=0.15, peak_electrons=10,  variance=0.05)),
])


def _level_params(level, **kwargs):
    """Fixed ranges for one named severity; kwargs override."""
    p = DEGRADATION_LEVELS[level]
    params = dict(peak_electrons=(p['peak_electrons'],) * 2,
                  variance=(p['variance'],) * 2,
                  compression_factor=(p['compression_factor'],) * 2,
                  mean_shift=(0.0, 0.0))
    params.update(kwargs)
    return params


class PoissonNoise3D(Augment):
    """Poisson noise augmentation for 3D volumes.

    Simulates shot noise from electron counting in EM imaging.
    Applies the same noise level uniformly across the entire 3D volume.

    Args:
        peak_electrons (tuple): Range (min, max) to sample effective electron
            count at max intensity. Lower = more noise. Typical EM: 50-500.
        skip (float): Probability of skipping the augmentation.
        invert (bool): Count electrons on 1 - I (FIB-SEM's inverted contrast).
    """
    def __init__(self, peak_electrons=(50, 200), skip=0.3, invert=False):
        self.peak_electrons = peak_electrons
        self.invert = invert
        self.skip = np.clip(skip, 0, 1)
        self.do_aug = False
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        self.do_aug = np.random.rand() > self.skip
        self.imgs = self._validate(spec, imgs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.do_aug:
            perturb = PoissonNoise(self.peak_electrons, invert=self.invert)
            for k in self.imgs:
                perturb(sample[k])
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'peak_electrons={}, '.format(self.peak_electrons)
        format_string += 'skip={:.2f}'.format(self.skip)
        format_string += ')'
        return format_string

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        return imgs


class PoissonNoise2D(Section):
    """Poisson noise applied per z-slice.

    Each z-slice can receive a different noise realization,
    simulating spatially varying shot noise.
    """
    def __init__(self, peak_electrons=(50, 200), prob=1, invert=False, **kwargs):
        super(PoissonNoise2D, self).__init__(
            PoissonNoise, prob=prob, **kwargs
        )
        self.params = dict(peak_electrons=peak_electrons, invert=invert)


class PoissonNoiseMixed(Blend):
    """Half 2D & half 3D Poisson noise."""
    def __init__(self, **kwargs):
        augments = [
            PoissonNoise2D(**kwargs),
            PoissonNoise3D(**kwargs)
        ]
        super(PoissonNoiseMixed, self).__init__(augments)


class GaussianNoise3D(Augment):
    """Gaussian noise augmentation for 3D volumes.

    Simulates electronic readout noise in EM imaging.
    Applies the same noise level uniformly across the entire 3D volume.

    Args:
        variance (tuple): Range (min, max) to sample noise variance from.
            Typical range: 0.001 to 0.05.
        skip (float): Probability of skipping the augmentation.
    """
    def __init__(self, variance=(0.005, 0.02), skip=0.3):
        self.variance = variance
        self.skip = np.clip(skip, 0, 1)
        self.do_aug = False
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        self.do_aug = np.random.rand() > self.skip
        self.imgs = self._validate(spec, imgs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.do_aug:
            perturb = GaussianNoise(self.variance)
            for k in self.imgs:
                perturb(sample[k])
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'variance={}, '.format(self.variance)
        format_string += 'skip={:.2f}'.format(self.skip)
        format_string += ')'
        return format_string

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        return imgs


class GaussianNoise2D(Section):
    """Gaussian noise applied per z-slice.

    Each z-slice can receive a different noise realization,
    simulating spatially varying readout noise.
    """
    def __init__(self, variance=(0.005, 0.02), prob=1, **kwargs):
        super(GaussianNoise2D, self).__init__(
            GaussianNoise, prob=prob, **kwargs
        )
        self.params = dict(variance=variance)


class GaussianNoiseMixed(Blend):
    """Half 2D & half 3D Gaussian noise."""
    def __init__(self, **kwargs):
        augments = [
            GaussianNoise2D(**kwargs),
            GaussianNoise3D(**kwargs)
        ]
        super(GaussianNoiseMixed, self).__init__(augments)


class ImageDegradation(Compose):
    """Composed image degradation pipeline.

    Applies degradation in physically-motivated order:
    1. Poisson noise (shot noise from electron counting)
    2. Gaussian noise (electronic readout noise)
    3. Contrast compression (reduced SNR/contrast)

    Based on Sardhara et al. (2022) noise model for EM imaging.

    Args:
        peak_electrons (tuple): Poisson noise parameter.
        variance (tuple): Gaussian noise parameter.
        compression_factor (tuple): Contrast compression parameter.
        mean_shift (tuple): Contrast compression mean shift.
        use_local_mean (bool): Use local mean for contrast compression.
        skip_poisson (float): Skip probability for Poisson noise.
        skip_gaussian (float): Skip probability for Gaussian noise.
        skip_contrast (float): Skip probability for contrast compression.
        invert (bool): Poisson noise on 1 - I (FIB-SEM's inverted contrast).
    """
    def __init__(self,
                 peak_electrons=(50, 200),
                 variance=(0.005, 0.02),
                 compression_factor=(0.4, 0.7),
                 mean_shift=(-0.1, 0.1),
                 use_local_mean=True,
                 skip_poisson=0.0,
                 skip_gaussian=0.0,
                 skip_contrast=0.0,
                 invert=False):
        augments = [
            PoissonNoise3D(peak_electrons=peak_electrons, skip=skip_poisson,
                           invert=invert),
            GaussianNoise3D(variance=variance, skip=skip_gaussian),
            ContrastCompression3D(
                compression_factor=compression_factor,
                mean_shift=mean_shift,
                use_local_mean=use_local_mean,
                skip=skip_contrast
            )
        ]
        super(ImageDegradation, self).__init__(augments)

    @classmethod
    def at_level(cls, level, **kwargs):
        """One named severity of ``DEGRADATION_LEVELS``, every draw the same."""
        return cls(**_level_params(level, **kwargs))


class ImageDegradation3D(ImageDegradation):
    """Alias for ImageDegradation (3D version)."""
    pass


class ImageDegradation2D(Compose):
    """Composed image degradation pipeline with per-slice noise.

    Same as ImageDegradation but applies noise per z-slice.
    """
    def __init__(self,
                 peak_electrons=(50, 200),
                 variance=(0.005, 0.02),
                 compression_factor=(0.4, 0.7),
                 mean_shift=(-0.1, 0.1),
                 use_local_mean=True,
                 prob=1.0,
                 skip_poisson=0.0,
                 skip_gaussian=0.0,
                 skip_contrast=0.0,
                 invert=False):
        augments = [
            PoissonNoise2D(peak_electrons=peak_electrons, prob=prob,
                          skip=skip_poisson, invert=invert),
            GaussianNoise2D(variance=variance, prob=prob, skip=skip_gaussian),
            ContrastCompression2D(
                compression_factor=compression_factor,
                mean_shift=mean_shift,
                use_local_mean=use_local_mean,
                prob=prob,
                skip=skip_contrast
            )
        ]
        super(ImageDegradation2D, self).__init__(augments)

    @classmethod
    def at_level(cls, level, **kwargs):
        """One named severity of ``DEGRADATION_LEVELS``, every draw the same."""
        return cls(**_level_params(level, **kwargs))
