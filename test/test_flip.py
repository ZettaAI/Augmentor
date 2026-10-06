"""FlipRotate covers 16 transforms and FlipRotateIsotropic 48, each uniformly.

A cube has 48 symmetries: 8 sign patterns times 6 orderings of its axes.
FlipRotateIsotropic used to draw three flips and three independent
transpositions -- 64 outcomes, 48 distinct transforms, 16 of them twice as
likely -- which these tests would have caught.

    python setup.py build_ext --inplace     # the package has a Cython extension
    pytest test/test_flip.py

The root conftest.py makes these import the checkout, not an installed copy.
"""
import collections
import itertools

import numpy as np
import pytest

from augmentor.flip import Flip, FlipRotate, FlipRotateIsotropic, Permute


def _probe():
    """A volume no symmetry of the cube maps onto itself."""
    return np.arange(27, dtype=np.float32).reshape(1, 3, 3, 3)


def _draw(aug, spec):
    """One draw: prepare on the output spec, then run on a matching sample."""
    in_spec = aug.prepare(dict(spec))
    sample = {k: np.arange(np.prod(v), dtype=np.float32).reshape(v)
              for k, v in in_spec.items()}
    return in_spec, aug(sample)


def _counts(aug, n, seed=0):
    np.random.seed(seed)
    spec = dict(input=(1, 3, 3, 3))
    counts = collections.Counter()
    zdest = collections.Counter()
    for _ in range(n):
        _, out = _draw(aug, spec)
        counts[out['input'].tobytes()] += 1
        # Where the first voxel's z-neighbour went tells where z went.
        delta = np.argwhere(out['input'] == 9)[0] - np.argwhere(out['input'] == 0)[0]
        zdest[int(np.flatnonzero(delta)[0]) - 1] += 1
    return counts, zdest


def test_permute_covers_the_six_orderings():
    assert sorted(Permute.PERMS) == sorted(itertools.permutations((1, 2, 3)))


def test_flip_rotate_is_16_uniform_and_keeps_z():
    counts, zdest = _counts(FlipRotate(), 16000)
    assert len(counts) == 16
    assert min(counts.values()) > 850 and max(counts.values()) < 1150
    assert set(zdest) == {0}


def test_flip_rotate_isotropic_is_48_uniform():
    counts, zdest = _counts(FlipRotateIsotropic(), 48000)
    assert len(counts) == 48
    # Expected 1000 each, sd about 31. The old sampler put 16 transforms at
    # 1500 and 32 at 750.
    assert min(counts.values()) > 850 and max(counts.values()) < 1150
    # The original z axis lands on each axis a third of the time; it used to
    # be a quarter, a half and a quarter.
    assert set(zdest) == {0, 1, 2}
    assert all(abs(v / 48000 - 1 / 3) < 0.02 for v in zdest.values())


def test_every_symmetry_is_one_flip_pattern_then_one_ordering():
    """The 8 x 6 parametrization hits each of the 48 exactly once."""
    x = _probe()
    seen = set()
    for bits in itertools.product([0, 1], repeat=3):
        y = x
        for bit, axis in zip(bits, (-1, -2, -3)):
            if bit:
                y = np.flip(y, axis)
        for perm in Permute.PERMS:
            seen.add(np.transpose(y, (0,) + perm).tobytes())
    assert len(seen) == 48


@pytest.mark.parametrize("seed", range(12))
def test_non_cubic_specs_come_out_in_the_requested_shape(seed):
    """prepare has to ask for the input with the inverse permutation, which
    differs from the permutation itself for the two cyclic orderings."""
    np.random.seed(seed)
    spec = dict(input=(1, 4, 5, 6), label=(3, 2, 3, 4), mask=(2, 3, 4))
    aug = FlipRotateIsotropic()
    in_spec, out = _draw(aug, spec)
    for k, v in spec.items():
        assert out[k].shape[-3:] == tuple(v[-3:]), (k, in_spec[k], out[k].shape)
        assert sorted(in_spec[k][-3:]) == sorted(v[-3:])


def _centre_crop(v, shape):
    lo = [(a - b) // 2 for a, b in zip(v.shape[-3:], shape[-3:])]
    return v[..., lo[0]:lo[0] + shape[-3], lo[1]:lo[1] + shape[-2], lo[2]:lo[2] + shape[-1]]


@pytest.mark.parametrize("seed", range(24))
def test_keys_of_different_sizes_stay_registered(seed):
    """A sample holds an image and a smaller label window read about one
    centre. After the transform the label still has to be the centre of the
    image, or the two no longer show the same place."""
    np.random.seed(seed)
    spec = dict(big=(1, 8, 10, 12), small=(1, 4, 6, 8))
    aug = FlipRotateIsotropic()
    in_spec = aug.prepare(dict(spec))
    big = np.random.rand(*in_spec['big']).astype(np.float32)
    sample = dict(big=big, small=_centre_crop(big, in_spec['small']).copy())
    out = aug(sample)
    assert out['big'].shape == spec['big'] and out['small'].shape == spec['small']
    assert np.array_equal(out['small'], _centre_crop(out['big'], spec['small']))
    # ... and the 24 seeds do move the data: this is not the identity passing.
    assert seed != 0 or not np.array_equal(out['big'], big)


def test_flip_alone_is_unchanged():
    np.random.seed(0)
    aug = Flip(axis=-3, prob=1)
    _, out = _draw(aug, dict(input=(1, 3, 3, 3)))
    assert np.array_equal(out['input'], np.flip(_probe(), -3))
