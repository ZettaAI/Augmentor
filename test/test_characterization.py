"""Pins what each augmentation DeepEM uses computes today, at a fixed seed.

Of the 48 names this package exports, DeepEM's recipes use 23; only the flips
had a test. These tests record the rest as they are -- they say nothing about
whether the behaviour is right, only that it has not moved. Each case is built
the way the recipes most often build it, driven the way DataProvider3 drives
it (`prepare(spec, imgs=, segs=)`, then call on a sample of the prepared
size), and reduced to a fingerprint in characterization.json:

    in_spec   the size `prepare` asked for, per key
    shape     of each output
    dtype     of each output
    blocks    mean of each output over a 3x4x4 grid of blocks

Block means are compared with a tolerance, so a different numpy or scipy build
does not turn the pins red, and a real change shows up in the blocks it touched.

    python setup.py build_ext --inplace
    pytest test/test_characterization.py
    python test/test_characterization.py --update   # after a deliberate change
"""
import json
import os
import random
import sys

import imgaug
import numpy as np
import pytest

from augmentor import *

PINS = os.path.join(os.path.dirname(__file__), 'characterization.json')
SEEDS = (0, 1, 2)
IMGS = ['input']
SEGS = ['segmentation']
PATCH = (12, 64, 64)
CUBE = (16, 16, 16)

# name -> (factory, output patch size). Arguments follow DeepEM's recipes, with
# `skip` zeroed so that every draw exercises the augmentation.
CASES = {
    'Compose': (lambda: Compose([
        MixedGrayscale2D(contrast_factor=0.5, brightness_factor=0.5, prob=1, skip=0),
        MixedBlurrySection(maxsec=7),
        FlipRotate(),
    ]), PATCH),
    'Blend': (lambda: Blend([
        Misalign((0, 15), margin=1),
        SlipMisalign((0, 15), interp=True, margin=1),
    ], props=[0.7, 0.3]), PATCH),
    'Label': (lambda: Label(targets=SEGS), PATCH),
    'Warp': (lambda: Warp(skip=0, do_twist=False, rot_max=45.0, scale_max=1.1), PATCH),
    'FlipRotate': (lambda: FlipRotate(), PATCH),
    'FlipRotateIsotropic': (lambda: FlipRotateIsotropic(), CUBE),
    'MixedBlurrySection': (lambda: MixedBlurrySection(maxsec=7), PATCH),
    'FillBox': (lambda: FillBox(dims=(5, 25), margin=(1, 5, 5), density=0.3, skip=0), PATCH),
    'NoiseBox': (lambda: NoiseBox(sigma=(1, 3), dims=(5, 25), margin=(1, 5, 5),
                                  density=0.3, skip=0), PATCH),
    'MixedGrayscale2D': (lambda: MixedGrayscale2D(contrast_factor=0.5, brightness_factor=0.5,
                                                  prob=1, skip=0), PATCH),
    'Grayscale3D': (lambda: Grayscale3D(skip=0), PATCH),
    'Misalign': (lambda: Misalign((0, 15), margin=1), PATCH),
    'SlipMisalign': (lambda: SlipMisalign((0, 15), interp=True, margin=1), PATCH),
    'MisalignPlusMissing': (lambda: MisalignPlusMissing((3, 15), value=0, random=False), PATCH),
    'MissingSection': (lambda: MissingSection(maxsec=7, individual=False, value=0,
                                              random=True), PATCH),
    'MixedMissingSection': (lambda: MixedMissingSection(maxsec=7, individual=True, value=0,
                                                        random=False), PATCH),
    'LostSection': (lambda: LostSection(1), PATCH),
    'LostPlusMissing': (lambda: LostPlusMissing(value=0, random=False), PATCH),
    'SectionGap': (lambda: SectionGap(num_secs=3, masked=True), PATCH),
    'Border': (lambda: Border(targets=SEGS), PATCH),
    'AdditiveGaussianNoise': (lambda: AdditiveGaussianNoise(sigma=(0.01, 0.1),
                                                            per_channel=False), PATCH),
    'ImageDegradation': (lambda: ImageDegradation(), PATCH),
    'ImageDegradation2D': (lambda: ImageDegradation2D(), PATCH),
}


def _sample(in_spec):
    """A sample of the prepared size whose values encode their own position.

    The image is a ramp along each axis plus fixed texture, in [0, 1]; the
    segmentation is a grid of blocks with a distinct id each and a background
    gap between them, so a shift, flip or dropped section moves the block means.
    """
    texture = np.random.default_rng(0)  # never the global stream under test
    sample = dict()
    for key, shape in in_spec.items():
        z, y, x = np.meshgrid(*[np.arange(n) for n in shape[-3:]], indexing='ij')
        if key in IMGS:
            ramp = 0.4 * z / shape[-3] + 0.25 * y / shape[-2] + 0.15 * x / shape[-1]
            data = ramp + 0.2 * texture.random(shape[-3:])
            sample[key] = data[None].astype(np.float32)
        elif key in SEGS:
            ids = 1 + (z // 4) * 100 + (y // 16) * 10 + (x // 16)
            ids[(y % 16 == 0) | (x % 16 == 0)] = 0
            sample[key] = ids[None].astype(np.uint32)
        else:
            sample[key] = np.ones((1,) + tuple(shape[-3:]), dtype=np.uint8)
    return sample


def _blocks(data):
    data = np.asarray(data, dtype=np.float64).reshape((-1,) + data.shape[-3:])[0]
    return [
        round(float(c.mean()), 6)
        for a in np.array_split(data, 3, axis=0)
        for b in np.array_split(a, 4, axis=1)
        for c in np.array_split(b, 4, axis=2)
    ]


def fingerprint(name, seed):
    factory, patch = CASES[name]
    np.random.seed(seed)
    random.seed(seed)
    imgaug.seed(seed)
    aug = factory()
    spec = {k: (1,) + patch for k in IMGS + SEGS + ['segmentation_mask']}
    in_spec = aug.prepare(dict(spec), imgs=list(IMGS), segs=list(SEGS))
    out = aug(_sample(in_spec))
    return dict(
        in_spec={k: [int(n) for n in in_spec[k]] for k in sorted(in_spec)},
        shape={k: list(out[k].shape) for k in sorted(out)},
        dtype={k: str(out[k].dtype) for k in sorted(out)},
        blocks={k: _blocks(out[k]) for k in sorted(out)},
    )


def _load():
    with open(PINS) as f:
        return json.load(f)


def test_cases_cover_what_deepem_uses():
    used = {
        'Compose', 'Label', 'Warp', 'FlipRotate', 'MixedBlurrySection', 'FillBox',
        'Blend', 'MixedGrayscale2D', 'Misalign', 'MixedMissingSection', 'SlipMisalign',
        'NoiseBox', 'Border', 'LostPlusMissing', 'LostSection', 'MisalignPlusMissing',
        'MissingSection', 'FlipRotateIsotropic', 'SectionGap', 'Grayscale3D',
        'AdditiveGaussianNoise', 'ImageDegradation', 'ImageDegradation2D',
    }
    assert set(CASES) == used
    assert set(_load()) == {f'{name}/{seed}' for name in CASES for seed in SEEDS}


@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('name', list(CASES))
def test_pinned(name, seed):
    pin = _load()[f'{name}/{seed}']
    got = fingerprint(name, seed)
    assert got['in_spec'] == pin['in_spec']
    assert got['shape'] == pin['shape']
    assert got['dtype'] == pin['dtype']
    for key, blocks in pin['blocks'].items():
        np.testing.assert_allclose(got['blocks'][key], blocks, atol=1e-5, err_msg=key)


@pytest.mark.parametrize('name', list(CASES))
def test_seed_determines_output(name):
    assert fingerprint(name, 0) == fingerprint(name, 0)


if __name__ == '__main__':
    if sys.argv[1:] != ['--update']:
        sys.exit(__doc__)
    pins = {f'{name}/{seed}': fingerprint(name, seed) for name in CASES for seed in SEEDS}
    with open(PINS, 'w') as f:
        f.write('{\n' + ',\n'.join(
            f'{json.dumps(k)}: {json.dumps(v)}' for k, v in pins.items()) + '\n}\n')
    print(f'wrote {len(pins)} pins to {PINS}')
