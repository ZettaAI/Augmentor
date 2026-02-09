import math
import numpy as np
from scipy.ndimage import map_coordinates

from .augment import Augment
from .geometry.box import Box

from pyfastnoiselite.pyfastnoiselite import (
    FastNoiseLite,
    FractalType,
    NoiseType,
)


class SimplexNoiseWarpField(Augment):
    """Non-rigid spatial deformation using OpenSimplex2 noise displacement fields.

    Generates a smooth 2D displacement field (dx, dy) from OpenSimplex2 noise
    and applies it to the sample. Displacements are always in-plane (Y, X).
    Uses pull notation: each output pixel is sampled from (x + dx, y + dy)
    in the input.

    By default the same field is applied to every Z-slice. When
    ``z_anisotropy`` is set, 3D noise is used instead so the field varies
    smoothly across Z while remaining a 2D displacement per slice.

    Args:
        max_displacement: Maximum pixel displacement in any direction. The noise
            output ([-1, 1]) is scaled by this value. Larger values produce
            stronger deformations. The input spec is expanded by
            2 * ceil(max_displacement) in Y and X to provide margin.
        frequency: Base spatial frequency of the noise. Controls the size of
            deformation features: lower values produce larger, smoother
            distortions (e.g. 0.005 ~ 200px wavelength), higher values produce
            finer, more local warping (e.g. 0.02 ~ 50px wavelength).
        octaves: Number of fractal noise layers (fBm). Each octave adds detail
            at progressively higher frequencies. 1 = single smooth layer,
            3-5 = natural-looking multi-scale deformation.
        lacunarity: Frequency multiplier between successive octaves. Controls
            how quickly the detail frequency increases. Default 2.0 means each
            octave doubles the frequency.
        gain: Amplitude multiplier between successive octaves (also called
            "persistence"). Controls how much each higher-frequency octave
            contributes. Default 0.5 means each octave has half the amplitude
            of the previous one. Lower values = smoother result, higher
            values = more high-frequency detail.
        skip: Probability of skipping this augmentation entirely for a given
            sample (0.0 = always apply, 1.0 = never apply).
        z_anisotropy: Ratio of XY resolution to Z resolution (e.g. 8.0/45.0
            for 8x8x45 nm voxels). When set, 3D noise is used and Z
            coordinates are scaled by this factor so the field transitions
            smoothly between slices at a physically consistent rate. When
            None (default), a single 2D field is shared across all Z-slices.
    """

    def __init__(
        self,
        max_displacement=16.0,
        frequency=0.01,
        octaves=3,
        lacunarity=2.0,
        gain=0.5,
        skip=0.3,
        z_anisotropy=None,
    ):
        self.max_displacement = max_displacement
        self.frequency = frequency
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.gain = gain
        self.skip = np.clip(skip, 0, 1)
        self.z_anisotropy = z_anisotropy
        self.do_warp = False
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        # Biased coin toss
        self.do_warp = np.random.rand() > self.skip
        if not self.do_warp:
            return dict(spec)

        self.imgs = self._validate(spec, imgs)

        # Save original spec.
        self.spec = dict(spec)

        # Compute the largest image size.
        box = Box((0,0,0), (0,0,0))
        for k, v in spec.items():
            box = box.merge(Box((0,0,0), v[-3:]))
        maxsz = tuple(box.size())

        # Simplex Warp padding (Y, X only; Z is unchanged in 2D mode)
        margin = int(math.ceil(self.max_displacement))
        padded_yx = tuple(x + 2 * margin for x in maxsz[-2:])

        # Increase tensor sizes
        ret = dict()
        for k, v in spec.items():
            if self.z_anisotropy is not None:
                # 3D: expand all dims to common size
                ret[k] = v[:-3] + (maxsz[0],) + padded_yx
            else:
                # 2D: keep each key's own Z, only pad Y/X
                ret[k] = v[:-3] + (v[-3],) + padded_yx
        return ret

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if not self.do_warp:
            return Augment.sort(sample)

        # Get expanded spatial dimensions from sample data
        first_key = next(iter(sample))
        v = sample[first_key]
        Z, H, W = v.shape[-3], v.shape[-2], v.shape[-1]

        # Generate displacement field(s)
        # dx, dy: (H, W) when 2D, (Z, H, W) when 3D
        dx, dy = self._generate_field(H, W, Z)

        # Base grid for sampling coordinates
        y_coords, x_coords = np.meshgrid(
            np.arange(H, dtype=np.float32),
            np.arange(W, dtype=np.float32),
            indexing="ij",
        )

        use_3d = self.z_anisotropy is not None

        # Pre-build warp coords per Z-slice (or one shared set)
        if use_3d:
            warp_coords_z = [
                np.array([y_coords + dy[z], x_coords + dx[z]])
                for z in range(Z)
            ]
        else:
            warp_coords = np.array([y_coords + dy, x_coords + dx])  # (2, H, W)

        # Apply warp to each key
        # Data is 4D after to_tensor(): (C, Z, Y, X)
        for k, v in sample.items():
            order = 1 if k in self.imgs else 0
            result = np.empty_like(v)
            for c in range(v.shape[0]):
                for z in range(v.shape[1]):
                    coords = warp_coords_z[z] if use_3d else warp_coords
                    result[c, z] = map_coordinates(
                        v[c, z], coords, order=order,
                        mode="constant", cval=0.0
                    )
            sample[k] = result

        # Center-crop each key back to its original spec size.
        # Keys may have different original (Z, Y, X) dimensions.
        # Offsets computed from actual sample dims (handles both 2D/3D modes).
        for k in sample:
            oz, oy, ox = self.spec[k][-3:]
            z_off = (sample[k].shape[-3] - oz) // 2
            y_off = (sample[k].shape[-2] - oy) // 2
            x_off = (sample[k].shape[-1] - ox) // 2
            # Copy to ensure contiguous memory layout (cf. Warp).
            sample[k] = sample[k][
                ...,
                z_off : z_off + oz,
                y_off : y_off + oy,
                x_off : x_off + ox,
            ].copy()

        return Augment.sort(sample)

    def _generate_field(self, H, W, Z=1):
        """Generate OpenSimplex2 displacement field.

        Args:
            H, W: Spatial dimensions.
            Z: Number of Z-slices. Only used when ``z_anisotropy`` is set.

        Returns:
            dx, dy: numpy arrays of shape (H, W) when 2D, or (Z, H, W) when
            3D (``z_anisotropy`` is set). Values are pixel displacements.
        """
        noise_gen = FastNoiseLite(seed=np.random.randint(0, 2**31))
        noise_gen.noise_type = NoiseType.NoiseType_OpenSimplex2
        noise_gen.frequency = self.frequency
        noise_gen.fractal_type = FractalType.FractalType_FBm
        noise_gen.fractal_octaves = self.octaves
        noise_gen.fractal_lacunarity = self.lacunarity
        noise_gen.fractal_gain = self.gain

        if self.z_anisotropy is not None:
            # 3D noise: field varies smoothly across Z
            zz, yy, xx = np.meshgrid(
                np.arange(Z, dtype=np.float32) * self.z_anisotropy,
                np.arange(H, dtype=np.float32),
                np.arange(W, dtype=np.float32),
                indexing="ij",
            )
            coords = np.array(
                [xx.flatten(), yy.flatten(), zz.flatten()], dtype=np.float32
            )
            shape = (Z, H, W)
        else:
            # 2D noise: single field shared across Z
            yy, xx = np.meshgrid(
                np.arange(H, dtype=np.float32),
                np.arange(W, dtype=np.float32),
                indexing="ij",
            )
            coords = np.array([xx.flatten(), yy.flatten()], dtype=np.float32)
            shape = (H, W)

        dx = (
            noise_gen.gen_from_coords(coords).reshape(shape)
            * self.max_displacement
        )
        coords_offset = coords + 10000.0  # easier than picking a new seed for dy
        dy = (
            noise_gen.gen_from_coords(coords_offset).reshape(shape)
            * self.max_displacement
        )
        return dx, dy

    def __repr__(self):
        parts = [
            f"max_displacement={self.max_displacement}",
            f"frequency={self.frequency}",
            f"octaves={self.octaves}",
            f"skip={self.skip:.2f}",
        ]
        if self.z_anisotropy is not None:
            parts.append(f"z_anisotropy={self.z_anisotropy}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        return imgs
