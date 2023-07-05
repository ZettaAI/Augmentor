from __future__ import print_function
import numpy as np

from datatools import create_border

from .augment import Augment


class Border(Augment):
    """
    Create borders
    """
    def __init__(self):
        self.segs = []

    def prepare(self, spec, segs=[], **kwargs):
        self.segs = segs
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        for k in self.segs:
            if k in sample:
                seg = sample[k][0,:,:,:].astype(np.uint32)
                sample[k] = create_border(seg).astype(np.uint32)
        return Augment.sort(Augment.to_tensor(sample))

    def __repr__(self):
        format_string = self.__class__.__name__ + '()'
        return format_string
