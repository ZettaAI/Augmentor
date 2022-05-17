import numpy as np
from scipy import ndimage


from .augment import Augment


__all__ = ['NormalView', 'TiltedView', 'TiltSeries']


class NormalView(object):
    """
    Normal view projection.
    """
    def __init__(self, num_sections, project_axis=-3):
        assert num_sections > 0
        self.num_sections = num_sections
        self.project_axis = project_axis

    def __call__(self, img):
        n = self.num_sections
        p = self.project_axis
        return NormalView.project(img, n, axis=p)

    @staticmethod
    def project(img, n, axis=-3):
        assert img.shape[axis] % n == 0
        idx = np.arange(0, img.shape[axis], n)[1:]
        arr = np.array_split(img, idx, axis=axis)
        prj = list(map(lambda x: np.average(x, axis=axis), arr))
        return np.stack(prj, axis=axis).astype(img.dtype)


class TiltedView(object):
    """
    Tilted view projection.
    """
    def __init__(self, num_sections, tilt_dir=1, tilt_axis=-1, project_axis=-3):
        self.num_sections = num_sections
        self.tilt_dir = tilt_dir
        self.tilt_axis = tilt_axis
        self.project_axis = project_axis

    def __call__(self, img):
        n = self.num_sections
        d = self.tilt_dir
        t = self.tilt_axis
        p = self.project_axis
        return TiltedView.project(img, n, d, t, p)
    
    @staticmethod
    def project(img, n, dir=1, axis_t=-1, axis_p=-3):
        assert img.shape[axis_p] % n == 0
        prj = np.copy(img)
        for i in range(1, n):
            idx = [slice(None)] * img.ndim
            idx[axis_p] = slice(i, img.shape[axis_p], n)
            idx = tuple(idx)
            prj[idx] = np.roll(img[idx], dir*i, axis=axis_t)
        return NormalView.project(prj, n, axis=axis_p)


class TiltSeries(Augment):
    """
    Tilt-series projections.
    """
    def __init__(self, num_sections):
        self.num_sections = num_sections

    def prepare(self, spec, **kwargs):
        pass

    def __call__(self, sample, **kwargs):
        pass

    def __repr__(self):
        format_string = self.__class__.__name__ + '()'
        return format_string
