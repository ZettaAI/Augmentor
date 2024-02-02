import numpy as np

from .augment import Augment
from .perturb import Grayscale


class Myelin(Augment):
    """Perturb myelin intensity.
    """
    def __init__(self, contrast_factor=0.3, brightness_factor=0.3, skip=0.3):
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor
        self.skip = np.clip(skip, 0, 1)
        self.do_aug = False
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        # Biased coin toss.
        self.do_aug = np.random.rand() > self.skip
        self.imgs = self._validate(spec, imgs)
        return dict(spec)
    
    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.do_aug and "myelin_aug" in sample:
            idx = sample["myelin_aug"] > 0
            perturb = Grayscale(self.contrast_factor, self.brightness_factor)
            for k in self.imgs:
                img = np.copy(sample[k])
                perturb(img)
                sample[k][idx] = img[idx]
        return Augment.sort(sample)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'contrast_factor={0}, '.format(self.contrast_factor)
        format_string += 'brightness_factor={0}, '.format(self.brightness_factor)
        format_string += 'skip={:.2f}'.format(self.skip)
        format_string += ')'
        return format_string

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        return imgs
