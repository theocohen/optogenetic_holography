import copy

import cv2
import torch
import matplotlib.pyplot as plt
plt.style.use('dark_background')

#from optogenetic_holography.optics.wavefront import WavefrontInterface


class Wavefront:
    def __init__(self, wavelength, pixel_pitch, resolution):
        self.u = torch.ones((1,1,) + resolution, dtype=torch.complex128)
        self.resolution = resolution
        self.wavelength = wavelength
        self.pixel_pitch = pixel_pitch

    @classmethod
    def from_image(cls, path, wavelength, pixel_pitch, intensity=True, scale_intensity=1):
        img = cv2.imread(path, 0)
        wf = Wavefront(wavelength, pixel_pitch, img.shape)
        if intensity:
            wf.intensity = torch.tensor(img / img.max() * scale_intensity)
        else:
            wf.phase = torch.tensor(img)
        return wf

    @property
    def amplitude(self):
        return self.u.abs()

    @amplitude.setter
    def amplitude(self, new_amplitude):
        self.u = torch.polar(new_amplitude.float(), self.phase.float())

    @property
    def phase(self):
        return self.u.angle()

    @phase.setter
    def phase(self, new_phase):
        self.u = torch.polar(self.amplitude.float(), new_phase)

    @property
    def intensity(self):
        return Wavefront.to_numpy(torch.square(self.amplitude))

    @intensity.setter
    def intensity(self, new_intensity):
        self.amplitude = torch.sqrt(new_intensity)

    @property
    def total_intensity(self):
        return Wavefront.to_numpy(torch.sum(torch.square(self.amplitude)))

    def replace_phase(self, target_amplitude):
        self.u = torch.polar(target_amplitude, self.phase)

    @classmethod
    def to_numpy(cls, tt):
        return tt.squeeze().cpu().detach().numpy()

    def assert_equal(self, other_field):
        return torch.allclose(self.u, other_field.u)

    def plot(self, fig_options=None, **kwargs):
        if 'intensity' in kwargs:
            options = kwargs['intensity']
            img = self.intensity / self.intensity.max() if options['normalize'] else self.intensity
        elif 'phase' in kwargs:
            img = Wavefront.to_numpy(self.phase)
            options = kwargs['phase']

        plt.imshow(img, cmap='gray')
        plt.xticks([]), plt.yticks([])
        if options['save']: plt.savefig(options['path'] + options['title'] + '.jpg', bbox_inches="tight", pad_inches = 0)
        plt.colorbar()
        plt.title(options['title'])
        plt.show()
        plt.close()

    def copy(self, copy_wf=False):
        return copy.deepcopy(self) if copy_wf else Wavefront(self.wavelength, self.pixel_pitch, self.resolution)