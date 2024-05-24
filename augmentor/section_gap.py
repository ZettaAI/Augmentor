from .augment import Augment


class SectionGap(Augment):
    """Section gap.

    Args:        
    """
    def __init__(self, num_secs=3, masked=False):
        assert num_secs > 0
        self.num_secs = num_secs
        self.masked = masked
        self.zlocs = {}
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        # Validate the inputs
        self._validate(spec, imgs)

        # Determine the target keys to process
        targets = list(imgs)
        if self.masked:
            targets += [key for key in spec.keys() if key.endswith("_mask")]

        # Calculate the dimensions and find the minimum depth
        zdims = {key: shape[-3] for key, shape in spec.items() if key in targets}
        zmin = min(zdims.values())
        assert zmin > self.num_secs

        # Calculate the starting z location
        zstart = (zmin - self.num_secs) // 2

        # Determine the z locations for each target
        self.zlocs = {key: zstart + (zdim - zmin) // 2 for key, zdim in zdims.items()}

        # Store the image list
        self.imgs = imgs
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        for k, zloc in self.zlocs.items():
            sample[k][..., zloc : zloc + self.num_secs, :, :] = 0
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'num_secs={}, '.format(self.num_secs)
        format_string += 'masked={}'.format(self.masked)
        format_string += ')'
        return format_string

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
