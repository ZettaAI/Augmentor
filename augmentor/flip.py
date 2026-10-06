from __future__ import print_function
import numpy as np

from .augment import Augment, Compose


__all__ = ['Flip', 'Transpose', 'Permute', 'FlipRotate', 'FlipRotateIsotropic']


class Flip(Augment):
    """Random flip.

    Args:
        axis (int):
        prob (float, optional):
    """
    def __init__(self, axis, prob=0.5):
        self.axis = axis
        self.prob = np.clip(prob, 0, 1)
        self.do_aug = False

    def prepare(self, spec, **kwargs):
        # Biased coin toss
        self.do_aug = np.random.rand() < self.prob
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.do_aug:
            for k, v in sample.items():
                # Prevent potential negative stride issues by copying.
                sample[k] = np.copy(np.flip(v, self.axis))
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'axis={}, '.format(self.axis)
        format_string += 'prob={:.3f}'.format(self.prob)
        format_string += ')'
        return format_string


class Transpose(Augment):
    """Random transpose.

    Args:
        axes (list of int, optional):
        prob (float, optional):
    """
    def __init__(self, axes=None, prob=0.5):
        assert (axes is None) or (len(axes)==4)
        self.axes = axes
        self.prob = np.clip(prob, 0, 1)
        self.do_aug = False

    def prepare(self, spec, **kwargs):
        spec = dict(spec)
        # Biased coin toss
        self.do_aug = np.random.rand() < self.prob
        if (not self.do_aug) or (self.axes is None):
            return spec
        for k, v in spec.items():
            assert len(v)==3 or len(v)==4
            offset = 1 if len(v)==3 else 0
            spec[k] = tuple(v[:-3]) + tuple(v[x - offset] for x in self.axes[-3:])
        return spec

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.do_aug:
            for k, v in sample.items():
                # Prevent potential negative stride issues by copying.
                sample[k] = np.copy(np.transpose(v, self.axes))
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'axes={}, '.format(self.axes)
        format_string += 'prob={:.3f}'.format(self.prob)
        format_string += ')'
        return format_string


class FlipRotate(Compose):
    def __init__(self):
        augs = [
            Flip(axis=-1),
            Flip(axis=-2),
            Flip(axis=-3),
            Transpose(axes=[0,1,3,2])
        ]
        super(FlipRotate, self).__init__(augs)


class Permute(Augment):
    """Random permutation of the three spatial axes.

    One of the six orderings of (z, y, x), drawn uniformly, the identity
    included. Unlike Transpose, which applies one fixed permutation or
    nothing, this covers every ordering with equal probability.
    """
    PERMS = [(1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1)]

    def __init__(self):
        self.axes = (0,1,2,3)

    def prepare(self, spec, **kwargs):
        spec = dict(spec)
        perm = Permute.PERMS[np.random.randint(len(Permute.PERMS))]
        self.axes = (0,) + perm
        # np.transpose puts input axis axes[i] at output position i, so the
        # input has to be asked for with the inverse permutation. Transpose
        # gets away without inverting because a swap is its own inverse; a
        # cycle is not.
        for k, v in spec.items():
            assert len(v)==3 or len(v)==4
            shape = [None] * 3
            for i, x in enumerate(perm):
                shape[x - 1] = v[len(v) - 3 + i]
            spec[k] = tuple(v[:-3]) + tuple(shape)
        return spec

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.axes != (0,1,2,3):
            for k, v in sample.items():
                # Prevent potential negative stride issues by copying.
                sample[k] = np.copy(np.transpose(v, self.axes))
        return Augment.sort(sample)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class FlipRotateIsotropic(Compose):
    """One of the 48 symmetries of the cube, drawn uniformly.

    Three independent flips give the 8 sign patterns and Permute gives the 6
    axis orderings; every symmetry is exactly one flip pattern followed by
    one ordering, so the product is uniform over all 48.

    This used to be three flips and three independent transpositions (xy, yz,
    zx): 64 equally likely outcomes, but only 48 distinct transforms, with 16
    of them twice as likely as the rest. The original z axis ended up on y
    half of the time and on z and x a quarter each.
    """
    def __init__(self):
        augs = [
            Flip(axis=-1),
            Flip(axis=-2),
            Flip(axis=-3),
            Permute()
        ]
        super(FlipRotateIsotropic, self).__init__(augs)
