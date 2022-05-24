import copy
import numpy as np
import skimage.measure as measure

from .augment import Augment


class SubsampleLabels(Augment):
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
        Augment.validate_spec(spec)
        spec = copy.deepcopy(spec)
        self.segs = self.__validate(spec, segs)
        
        # Update spec
        segs = list(self.segs) + [x + '_mask' for x in self.segs]
        for k in segs:
            if k not in segs:
                assert '_mask' in k
                continue
            # Shape
            s = spec[k]['shape']
            spec[k]['shape'] = s[:-3] + tuple(s[-3:] * self.factor)
            # Resolution
            r = spec[k]['resolution']
            spec[k]['resolution'] = tuple(r // self.factor)
        return spec

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if any(self.factor > 1):
            for k in self.segs:
                sample[k] = sample[k][self.slice]
                m = k + '_mask'
                if m in sample:
                    sample[m] = sample[m][self.slice]
        return Augment.sort(Augment.to_tensor(sample))

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'factor={self.factor}'
        format_string += ')'
        return format_string

    def __validate(self, spec, segs):
        assert len(segs) > 0
        assert all(k in spec for k in segs)
        return segs