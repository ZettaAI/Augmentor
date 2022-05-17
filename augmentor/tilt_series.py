import numpy as np
from scipy import ndimage

from .augment import Augment


__all__ = ['NormalView', 'TiltedView', 'SimpleTiltSeries', 'TiltSeries']


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
        assert (dir == 1) or (dir == -1)
        prj = np.copy(img)
        pivot = n//2
        for i in range(0, n):
            if i == pivot:
                continue
            idx = [slice(None)] * img.ndim
            idx[axis_p] = slice(i, img.shape[axis_p], n)
            idx = tuple(idx)
            prj[idx] = np.roll(img[idx], dir*(i - pivot), axis=axis_t)
        prj = NormalView.project(prj, n, axis=axis_p)
        
        # # Invalidate the first half
        # idx_l = [slice(None)] * img.ndim
        # idx_l[axis_t] = slice(0, pivot)
        # prj[tuple(idx_l)] = 0
        
        # # Invalidate the second half
        # idx_r = [slice(None)] * img.ndim
        # idx_r[axis_t] = slice(-pivot, None)        
        # prj[tuple(idx_r)] = 0

        return prj


def simple_projections(num_sections): 
    prjs = list()
    prjs.append(NormalView(num_sections))
    prjs.append(TiltedView(num_sections,  1, -1))  # y + 45
    prjs.append(TiltedView(num_sections, -1, -1))  # y - 45
    prjs.append(TiltedView(num_sections,  1, -2))  # x + 45
    prjs.append(TiltedView(num_sections, -1, -2))  # x - 45
    return prjs


class SimpleTiltSeries(object):
    """
    Tilt-series projections.
    """
    def __init__(self, num_sections):
        self.num_sections = num_sections
        self.prjs = simple_projections(num_sections)

    def __call__(self, img):
        return self.project(img)

    def project(self, img):
        assert img.ndim >= 3
        prjs = [prj(img) for prj in self.prjs]
        if img.ndim == 3:
            tilt_series = np.stack(prjs)
        else:
            tilt_series = np.concatenate(prjs, axis=-4)
        return tilt_series

        
class TiltSeries(Augment):
    """
    Tilt-series projections.
    """
    def __init__(self, num_sections):
        self.num_sections = num_sections
        self.pad = num_sections // 2
        self.sts = SimpleTiltSeries(num_sections)
        self.imgs = []        

    def prepare(self, spec, imgs=[], **kwargs):
        self.imgs = self.__validate(spec, imgs)
        pad = 2*self.pad
        # Update spec
        spec = dict(spec)
        for k in self.imgs:
            v = spec[k]
            spec[k] = v[:-2] + (v[-2]+pad, v[-1]+pad)
        return spec

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)        
        for k in self.imgs:
            img = sample[k]            
            assert img.shape[-4] == 1
            p = self.pad
            sample[k] = self.sts(img)[..., p:-p, p:-p]
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'num_sections={self.num_sections}'
        format_string += ')'
        return format_string

    def __validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        return imgs