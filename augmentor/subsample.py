import numpy as np

import skimage.measure as measure

from .augment import Augment, Compose


class Subsample(Augment):
    """
    Subsample.
    """
    def __init__(self, factor=(1,1,1)):
        self.segs = []
        self.factor = np.array(factor)
        self.start = self.factor//2
        slc = [slice(s, None, f) for s, f in zip(self.start, self.factor)]
        self.slice = tuple([slice(0, None)] + slc)

    def prepare(self, spec, segs=[], **kwargs):
        self.segs = self._validate(spec, segs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if any(self.factor > 1):
            for k in self.segs:
                sample[k] = sample[k][self.slice].astype(np.uint32)
                m = k + '_mask'
                if m in sample:
                    sample[m] = sample[m][self.slice].astype(np.uint8)
        return Augment.sort(Augment.to_tensor(sample))

    def __repr__(self):
        format_string = self.__class__.__name__ + '()'
        return format_string

    def _validate(self, spec, segs):
        assert len(segs) > 0
        assert all(k in spec for k in segs)
        return segs