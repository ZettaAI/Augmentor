import numpy as np
import skimage.measure as measure

from .augment import Augment


__all__ = ['CropLabels']


def crop_center_no_strict(v, size):
    assert v.ndim > 3
    assert len(size) == 3
    idx = [slice(None)] * (v.ndim - 3)
    for x, y in zip(v.shape[-3:], size[-3:]):
        if x > y:
            s = (x - y)//2
            idx.append(slice(s, s + y))
        else:
            idx.append(slice(None))
    return v[idx]


class CropLabels(Augment):
    """
    Crop labels.
    """
    def __init__(self, crop):
        self.crop = np.array(crop)
        self.segs = []        

    def prepare(self, spec, segs=[], **kwargs):
        self.segs = self.__validate(spec, segs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.crop is not None:
            for k in self.segs:
                v = sample[k]
                cropsz = (np.array(v.shape[-3:]) * self.crop).astype(int)
                sample[k] = crop_center_no_strict(v, cropsz)                
                m = k + '_mask'
                if m in sample:
                    v = sample[m]
                    cropsz = (np.array(v.shape[-3:]) * self.crop).astype(int)
                    sample[m] = crop_center_no_strict(v, cropsz)
        return Augment.sort(Augment.to_tensor(sample))

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'crop={self.crop}'
        format_string += ')'
        return format_string

    def __validate(self, spec, segs):
        assert len(segs) > 0
        assert all(k in spec for k in segs)
        return segs