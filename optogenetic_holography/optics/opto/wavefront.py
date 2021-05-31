import copy
import glob

import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('dark_background')

#from optogenetic_holography.optics.wavefront import WavefrontInterface


class Wavefront:
    def __init__(self, wavelength, pixel_pitch, resolution, depth=1):
        assert len(resolution) == 2
        self.resolution = resolution
        self.depth = depth
        self.u = torch.ones(self.shape, dtype=torch.complex128)
        self.wavelength = wavelength
        self.pixel_pitch = pixel_pitch

    @classmethod
    def from_images(cls, path, wavelength, pixel_pitch, intensity=True, scale_intensity=1):
        images = np.stack([cv2.imread(file, 0) for file in sorted(glob.glob(path))])
        wf = Wavefront(wavelength, pixel_pitch, images.shape[1:], depth=images.shape[0])
        if intensity:
            wf.intensity = torch.tensor(images * scale_intensity).double()
            wf.amplitude /= wf.amplitude.amax(dim=(1, 2, 3), keepdim=True)
        else:
            wf.phase = torch.tensor(images)
        return wf

    @property
    def shape(self):
        return (self.depth, 1,) + self.resolution

    @shape.setter
    def shape(self, shape):
        self.depth = shape[0]
        self.resolution = shape[:2]

    @property
    def amplitude(self):
        return self.u.abs()

    @property
    def scaled_amplitude(self):
        return self.amplitude / self.amplitude.amax(dim=(1,2,3), keepdim=True)

    @amplitude.setter
    def amplitude(self, new_amplitude):
        if len(new_amplitude.shape) < 4:
            new_amplitude = new_amplitude.reshape(self.shape)
        self.u = torch.polar(new_amplitude, self.phase)

    @property
    def phase(self):
        return self.u.angle()

    @phase.setter
    def phase(self, new_phase):
        self.u = torch.polar(self.amplitude, new_phase.reshape(self.shape))

    @property
    def intensity(self):
        return Wavefront.to_numpy(torch.square(self.amplitude))

    @intensity.setter
    def intensity(self, new_intensity):
        self.amplitude = torch.sqrt(new_intensity)

    @property
    def total_intensity(self):
        return Wavefront.to_numpy(torch.sum(torch.square(self.amplitude), (-2,-1)))

    @classmethod
    def to_numpy(cls, tt):
        return tt.squeeze(1).cpu().detach().numpy()

    def polar_to_rect(self, amp, phase):
        """from neural holo"""
        self.u = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase)).reshape(self.shape)

    def assert_equal(self, other_field, atol=1e-6):
        return torch.allclose(self.u, other_field.u, atol=atol)

    def plot(self, fig_options=None, **kwargs):
        if 'intensity' in kwargs:
            options = kwargs['intensity']
            img = self.intensity / self.intensity.max() if options['normalize'] else self.intensity
        elif 'phase' in kwargs:
            img = Wavefront.to_numpy(self.phase)
            options = kwargs['phase']

        for i in range(self.depth):
            plt.imshow(img[i], cmap='gray')
            plt.xticks([]), plt.yticks([])
            if options['save']: plt.savefig(options['path'] + options['title'] + str(i+1) + '.jpg', bbox_inches="tight", pad_inches = 0)
            plt.colorbar()
            plt.title(options['title'] + '_plane_' + str(i))
            plt.show()
            plt.close()

    def copy(self, copy_wf=False):
        return copy.deepcopy(self) if copy_wf else Wavefront(self.wavelength, self.pixel_pitch, self.resolution, depth=self.depth)

    def requires_grad(self):
        self.u.requires_grad_(True)