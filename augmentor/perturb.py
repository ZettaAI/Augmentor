from __future__ import print_function
import numpy as np
from scipy.ndimage.filters import gaussian_filter


class Perturb(object):
    """
    Callable class for in-place image perturbation.
    """
    def __init__(self):
        raise NotImplementedError

    def __call__(self, img):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Grayscale(Perturb):
    """Grayscale intensity perturbation."""
    def __init__(self, contrast_factor=0.3, brightness_factor=0.3):
        contrast_factor = np.clip(contrast_factor, 0, 2)
        brightness_factor = np.clip(brightness_factor, 0, 2)
        params = dict()
        params['contrast'] = 1 + (np.random.rand() - 0.5) * contrast_factor
        params['brightness'] = (np.random.rand() - 0.5) * brightness_factor
        params['gamma'] = (np.random.rand()*2 - 1)
        self.params = params

    def __call__(self, img):
        img *= self.params['contrast']
        img += self.params['brightness']
        np.clip(img, 0, 1, out=img)
        img **= 2.0**self.params['gamma']

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'contrast={:.2f}, '.format(self.params['contrast'])
        format_string += 'brightness={:.2f}, '.format(self.params['brightness'])
        format_string += 'gamma={:.2f}'.format(self.params['gamma'])
        format_string += ')'
        return format_string


class Fill(Perturb):
    """Fill with a scalar."""
    def __init__(self, value=0, random=False):
        value = np.clip(value, 0, 1)
        self.value = np.random.rand() if random else value

    def __call__(self, img):
        img[...] = self.value

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'value={:.3f}'.format(self.value)
        format_string += ')'
        return format_string


class Blur(Perturb):
    """Gaussian blurring."""
    def __init__(self, sigma=5.0, random=False):
        sigma = max(sigma, 0)
        self.sigma = np.random.rand()*sigma if random else sigma

    def __call__(self, img):
        gaussian_filter(img, sigma=self.sigma, output=img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'sigma={:.2f}'.format(self.sigma)
        format_string += ')'
        return format_string


class Blur3D(Perturb):
    """Gaussian blurring."""
    def __init__(self, sigma=(5.0,5.0,5.0), random=False):
        self.sigma = [np.random.rand()*s for s in sigma] if random else sigma

    def __call__(self, img):
        gaussian_filter(img, sigma=self.sigma, output=img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'sigma=({:.2f},{:.2f},{:.2f})'.format(*self.sigma)
        format_string += ')'
        return format_string


class Noise(Perturb):
    """Uniform noise + Gaussian blurring."""
    def __init__(self, sigma=(2,5)):
        assert len(sigma)==2
        self.sigma = tuple(max(s, 0) for s in sigma)

    def __call__(self, img):
        patch = (np.random.rand(*img.shape[-3:])).astype(img.dtype)
        s1 = self.sigma[0]
        gaussian_filter(patch, sigma=(0,s1,s1), output=patch)
        patch = (patch > 0.5).astype(img.dtype)
        s2 = self.sigma[1]
        gaussian_filter(patch, sigma=(0,s2,s2), output=patch)
        img[...,:,:,:] = patch

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'sigma={:}'.format(self.sigma)
        format_string += ')'
        return format_string


class PoissonNoise(Perturb):
    """Add Poisson noise to simulate primary electron variability.

    This augmentation simulates shot noise from electron counting in EM imaging.
    The noise level is controlled by `peak_electrons` - lower values produce
    more noise.

    Args:
        peak_electrons (tuple): Range (min, max) to sample effective electron
            count at max intensity. Lower = more noise. Typical EM: 50-500.
            Default is (50, 200).
    """
    def __init__(self, peak_electrons=(50, 200)):
        self.peak_electrons_range = peak_electrons
        self.peak_electrons = np.random.uniform(*peak_electrons)

    def __call__(self, img):
        # Scale to electron counts
        electrons = img * self.peak_electrons
        # Apply Poisson sampling
        noisy = np.random.poisson(electrons).astype(np.float64)
        # Scale back and store in-place
        img[...] = noisy / self.peak_electrons
        np.clip(img, 0, 1, out=img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'peak_electrons={:.1f}'.format(self.peak_electrons)
        format_string += ')'
        return format_string


class GaussianNoise(Perturb):
    """Add Gaussian noise to simulate electronic readout interference.

    This augmentation adds zero-mean Gaussian noise to simulate electronic
    readout noise in EM imaging (Sardhara et al., 2022).

    Args:
        variance (tuple): Range (min, max) to sample noise variance (σ²) from.
            Typical range: 0.001 to 0.05. Default is (0.005, 0.02).
    """
    def __init__(self, variance=(0.005, 0.02)):
        self.variance_range = variance
        self.variance = np.random.uniform(*variance)
        self.sigma = np.sqrt(self.variance)

    def __call__(self, img):
        noise = np.random.normal(0, self.sigma, img.shape)
        img += noise
        np.clip(img, 0, 1, out=img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'variance={:.4f}'.format(self.variance)
        format_string += ')'
        return format_string


class ContrastCompression(Perturb):
    """Compress intensity distribution toward the mean.

    This augmentation simulates low SNR/contrast regions by narrowing the
    intensity distribution. The transformation is:

        new_img = mean + (img - mean) * compression_factor

    where compression_factor < 1 narrows the histogram.

    Args:
        compression_factor (tuple): Range (min, max) to sample compression
            factor from. Values < 1 compress the distribution. Default is
            (0.4, 0.7) based on observed degradation in problematic regions.
        mean_shift (tuple): Range (min, max) to sample mean shift from.
            Applied as a fraction of the original mean. Default is (-0.1, 0.1).
        use_local_mean (bool): If True, compute mean from the input image.
            If False, use 0.5 as the assumed mean for normalized images.
    """
    def __init__(self, compression_factor=(0.4, 0.7), mean_shift=(-0.1, 0.1),
                 use_local_mean=True):
        self.compression_factor = compression_factor
        self.mean_shift = mean_shift
        self.use_local_mean = use_local_mean

        # Sample random parameters
        self.factor = np.random.uniform(*compression_factor)
        self.shift = np.random.uniform(*mean_shift)

    def __call__(self, img):
        # Compute mean
        if self.use_local_mean:
            mean = img.mean()
        else:
            mean = 0.5

        # Apply compression: new = mean + (img - mean) * factor
        img -= mean
        img *= self.factor
        img += mean

        # Apply mean shift
        img += self.shift * mean

        # Clip to valid range
        np.clip(img, 0, 1, out=img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'factor={:.3f}, '.format(self.factor)
        format_string += 'shift={:.3f}, '.format(self.shift)
        format_string += 'use_local_mean={}'.format(self.use_local_mean)
        format_string += ')'
        return format_string
