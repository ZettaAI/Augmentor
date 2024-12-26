from einops import rearrange, reduce
from .augment import Augment


__all__ = ['FoldToChannel', 'UnfoldFromChannel', 'DownsampleInput']


class FoldToChannel(Augment):
    def __init__(self, num_sections):
        self.num_sections = num_sections

    def prepare(self, spec, **kwargs):
        num = self.num_sections
        # Update spec
        spec = dict(spec)
        for k, v in spec.items():
            spec[k] = v[:-4] + (1, v[-3]*num, v[-2], v[-1])
        return spec

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        for k, v in sample.items():
            sample[k] = rearrange(v, "() (d c) h w -> c d h w ", c=self.num_sections)
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'num_sections={self.num_sections}'
        format_string += ')'
        return format_string


class UnfoldFromChannel(Augment):
    def __init__(self, num_sections):
        self.num_sections = num_sections
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        num = self.num_sections
        # Update spec
        self.imgs = self._validate(spec, imgs)
        spec = dict(spec)
        for k, v in spec.items():
            if k not in self.imgs:
                spec[k] = v[:-4] + (num, v[-3]//num, v[-2], v[-1])
        return spec

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        for k, v in sample.items():
            if k not in self.imgs:
                sample[k] = rearrange(v, "c d h w -> () (d c) h w ", c=self.num_sections)
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'num_sections={self.num_sections}'
        format_string += ')'
        return format_string

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        return imgs


class DownsampleInput(Augment):
    def __init__(self, num_sections):
        self.num_sections = num_sections
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        num = self.num_sections
        # Update spec
        self.imgs = self._validate(spec, imgs)
        spec = dict(spec)
        for k in self.imgs:
            v = spec[k]
            spec[k] = v[:-4] + (num, v[-3], v[-2], v[-1])
        return spec

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        for k in self.imgs:
            v = sample[k]
            sample[k] = reduce(v, "c d h w -> () d h w ", "mean")
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'num_sections={self.num_sections}'
        format_string += ')'
        return format_string

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        return imgs
