"""FIBSEMNoise adds the noise its parameters describe, and only that.

    python setup.py build_ext --inplace     # the package has a Cython extension
    pytest test/test_fibsem.py
"""
import numpy as np
import pytest

from augmentor.degradation import DEGRADATION_LEVELS, ImageDegradation, PoissonNoise3D
from augmentor.fibsem import FIBSEM_LEVELS, FIBSEMNoise, fibsem_noise


def _flat(value, shape=(1, 64, 64, 64)):
    return np.full(shape, value, dtype=np.float32)


def test_variance_follows_the_darkness_of_the_pixel():
    np.random.seed(0)
    for level in (0.3, 0.7):
        img = fibsem_noise(_flat(level), noise_var=0.001, dark_slope=0.004,
                           sigma_x=0, slice_gain=0)
        expected = 0.001 + 0.004 * (1 - level)
        assert img.var() == pytest.approx(expected, rel=0.05)


def test_not_inverted_the_variance_grows_with_brightness():
    np.random.seed(0)
    img = fibsem_noise(_flat(0.7), noise_var=0.001, dark_slope=0.004,
                       sigma_x=0, slice_gain=0, invert=False)
    assert img.var() == pytest.approx(0.001 + 0.004 * 0.7, rel=0.05)


def _lag1(img, axis):
    a = img - img.mean()
    return float((a * np.roll(a, 1, axis=axis)).mean() / a.var())


def test_noise_correlates_along_the_fast_axis_only():
    np.random.seed(0)
    img = fibsem_noise(_flat(0.5), noise_var=0.002, dark_slope=0,
                       sigma_x=0.8, slice_gain=0, fast_axis=-1)
    assert _lag1(img, -1) > 0.4
    assert abs(_lag1(img, -2)) < 0.02
    assert abs(_lag1(img, -3)) < 0.02
    # The kernel has unit energy: the variance is unchanged.
    assert img.var() == pytest.approx(0.002, rel=0.05)


def test_slice_gain_moves_whole_slices_of_the_inverted_signal():
    np.random.seed(0)
    img = fibsem_noise(_flat(0.4), noise_var=0, dark_slope=0, sigma_x=0,
                       slice_gain=0.05)
    per_slice = img[0].reshape(img.shape[1], -1)
    assert np.allclose(per_slice.std(axis=1), 0, atol=1e-6)
    # Inverted signal 0.6, so each slice moves by 0.6 * gain.
    assert per_slice[:, 0].std() == pytest.approx(0.6 * 0.05, rel=0.2)


def test_oversampling_averages_the_noise_down():
    np.random.seed(0)
    img = fibsem_noise(_flat(0.5), noise_var=0.004, dark_slope=0,
                       sigma_x=0, slice_gain=0, oversample=2)
    assert img.shape == (1, 64, 64, 64)
    assert img.var() == pytest.approx(0.004 / 8, rel=0.05)


def test_the_augment_draws_once_per_sample_and_honours_skip():
    np.random.seed(0)
    aug = FIBSEMNoise(skip=1.0)
    spec = dict(input=(1, 32, 32, 32), label=(1, 32, 32, 32))
    aug.prepare(dict(spec), imgs=['input'])
    sample = dict(input=_flat(0.5, (1, 32, 32, 32)),
                  label=_flat(0.5, (1, 32, 32, 32)))
    out = aug(sample)
    assert np.all(out['input'] == 0.5)

    aug = FIBSEMNoise(skip=0.0)
    aug.prepare(dict(spec), imgs=['input'])
    out = aug(dict(input=_flat(0.5, (1, 32, 32, 32)),
                   label=_flat(0.5, (1, 32, 32, 32))))
    assert out['input'].std() > 0
    assert np.all(out['label'] == 0.5)        # only the named images change
    assert 0 <= out['input'].min() and out['input'].max() <= 1


def _effective_snr(clean, noisy):
    return clean.var() / (noisy - clean).var()


def test_fibsem_levels_get_harder_in_order():
    np.random.seed(0)
    clean = np.random.rand(1, 32, 32, 32).astype(np.float32) * 0.6 + 0.2
    snrs = []
    for level in FIBSEM_LEVELS:
        aug = FIBSEMNoise.at_level(level)
        aug.prepare(dict(input=clean.shape), imgs=['input'])
        out = aug(dict(input=clean.copy()))['input']
        snrs.append(_effective_snr(clean, out))
    assert all(a > b for a, b in zip(snrs, snrs[1:]))


def test_image_degradation_levels_are_fixed_and_ordered():
    assert list(DEGRADATION_LEVELS)[0] == 'mild'
    assert list(DEGRADATION_LEVELS)[-1] == 'nightmare'
    aug = ImageDegradation.at_level('harsh')
    poisson, gaussian, contrast = aug.augments
    assert poisson.peak_electrons == (30, 30)
    assert gaussian.variance == (0.025, 0.025)
    assert contrast.compression_factor == (0.35, 0.35)


def test_inverted_poisson_noise_is_largest_in_the_dark():
    np.random.seed(0)
    spec = dict(input=(1, 32, 32, 32))
    for invert, noisier in ((False, 'bright'), (True, 'dark')):
        var = {}
        for name, level in (('dark', 0.2), ('bright', 0.8)):
            aug = PoissonNoise3D(peak_electrons=(50, 50), skip=0, invert=invert)
            aug.prepare(dict(spec), imgs=['input'])
            var[name] = aug(dict(input=_flat(level, (1, 32, 32, 32))))['input'].var()
        other = 'dark' if noisier == 'bright' else 'bright'
        assert var[noisier] > 2 * var[other]
