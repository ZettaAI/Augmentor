import copy
import numpy as np

from .augment import Augment


__all__ = ['LostSection', 'LostPlusMissing']


class LostSection(Augment):
    """Lost section augmentation.

    Args:
        nsec: number of consecutive lost sections.
        skip (float, optional): skip probability.

    TODO:
        Support for valid architecture.
    """
    def __init__(self, nsec, skip=0, **kwargs):
        self.nsec = max(nsec, 1)
        self.skip = np.clip(skip, 0, 1)
        self.zloc = {}

    def prepare(self, spec, **kwargs):
        Augment.validate_spec(spec)
        spec = copy.deepcopy(spec)

        # Biased coin toss
        if np.random.rand() < self.skip:
            self.zloc = {}
            return dict(spec)

        # Collect z-info.
        spec = copy.deepcopy(spec)
        zdims = {k: v['shape'][-3] for k, v in spec.items()}
        zres = {k: v['resolution'][-3] for k, v in spec.items()}
        rmax = max(zres.values())
        zscale = {k: rmax // v for k, v in zres.items()}
        zdims2 = {k: zdims[k] * v for v in zres.items()}

        # Pick a random section.
        zmin2 = min(zdims2.values())
        assert zmin2 % rmax == 0
        zmin = zmin2 // rmax - 1        
        zloc = (np.random.choice(zmin, 1, replace=False) + 1)[0]

        # Offset z-location.
        self.zslcs = {}
        for k, zdim in zdims.items():
            offset = (zdim - zmin) // 2
            zstart = (offset + zloc) * zscale[k]
            self.zslcs[k] = slice(zstart, zstart + zscale[k])

        # Update spec
        for k, v in spec.items():
            zslc = self.zslcs[k]
            step = zslc.stop - zslc.start
            new_z = v['shape'][-3] + (self.nsec * step)
            spec[k]['shape'] = v[:-3] + (new_z,) + v[-2:]
        
        return spec

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if len(self.zslcs) > 0:
            n = self.nsec
            for k, v in sample.items():
                zslc = self.zslcs[k]
                zloc = zslc.start
                step = zslc.stop - zslc.start

                # New tensor
                c, z, y, x = v.shape[-4:]
                w = np.zeros((c, z - n*step, y, x), dtype=v.dtype)

                # Non-missing part
                w[:,:zloc,:,:] = v[:,:zloc,:,:]
                w[:,zloc:,:,:] = v[:,zloc+n*step:,:,:]
                
                # Update sample
                sample[k] = w

        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'nsec={:}, '.format(self.nsec)
        format_string += 'skip={:.2f}, '.format(self.skip)
        format_string += ')'
        return format_string


class LostPlusMissing(LostSection):
    def __init__(self, skip=0, value=0, random=False):
        super(LostPlusMissing, self).__init__(2, skip=skip)
        self.value = value
        self.random = random
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        spec = super(LostPlusMissing, self).prepare(spec)
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        self.imgs = imgs
        return dict(spec)

    def __call__(self, sample, **kwargs):
        return self.augment(sample)

    def augment(self, sample):
        sample = Augment.to_tensor(sample)

        if len(self.zslcs) > 0:
            val = np.random.rand() if self.random else self.value
            assert self.nsec == 2
            n = self.nsec
            for k, v in sample.items():
                zslc = self.zslcs[k]
                zloc = zslc.start
                step = zslc.stop - zslc.start

                # New tensor
                c, z, y, x = v.shape[-4:]
                w = np.zeros((c, z - n*step, y, x), dtype=v.dtype)

                # Non-missing part
                w[:,:zloc,:,:] = v[:,:zloc,:,:]
                w[:,zloc+step:,:,:] = v[:,zloc+(n+1)*step:,:,:]

                # Missing part
                if k in self.imgs:
                    w[:,zslc,...] = val
                else:
                    src = slice(zslc.stop, zslc.stop + step)
                    w[:,zslc,...] = v[:,src,:,:]

                # Update sample
                sample[k] = w

        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'nsec={:}, '.format(self.nsec)
        format_string += 'skip={:.2f}, '.format(self.skip)
        format_string += 'value={:}, '.format(self.value)
        format_string += 'random={:}, '.format(self.random)
        format_string += ')'
        return format_string
