import cv2
import torch
import matplotlib.pyplot as plt
plt.style.use('dark_background')

#from optogenetic_holography.optics.wavefront import WavefrontInterface


class Wavefront:
    def __init__(self, wavelength, pixel_pitch, resolution):
        self.u = torch.ones(resolution, dtype=torch.complex128)
        self.resolution = resolution
        self.wavelength = wavelength
        self.pixel_pitch = pixel_pitch

    @classmethod
    def from_image(cls, path, wavelength, pixel_pitch, scale_intensity=1):
        img = cv2.imread(path, 0)
        wf = Wavefront(wavelength, pixel_pitch, img.shape)
        wf.u = torch.tensor(img / img.max() * scale_intensity, dtype=torch.complex128)
        return wf

    @property
    def amplitude(self):
        return self.u.abs()

    @amplitude.setter
    def amplitude(self, new_amplitude):
        self.u = torch.polar(new_amplitude, self.phase.double())

    @property
    def phase(self):
        return self.u.angle()

    @phase.setter
    def phase(self, new_phase):
        self.u = torch.polar(self.amplitude.float(), new_phase)

    @property
    def intensity(self):
        i = torch.square(self.amplitude)
        return i

    @property
    def total_intensity(self):
        return float(self._numpy(self.intensity.sum()))

    def replace_phase(self, target_amplitude):
        self.u = torch.polar(target_amplitude, self.phase)

    def _numpy(self, tt):
        return tt.cpu().detach().numpy()

    def assert_equal(self, other_field):
        return torch.allclose(self.u, other_field.u)

    def plot(self, fig_options=None, **kwargs):
        options = kwargs['intensity']
        i = self.intensity / self.intensity.max() if options['normalize'] else self.intensity
        plt.imshow(i, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.colorbar()
        plt.title(options['title'])
        if options['save']: plt.savefig(options['path'] + options['title'] + '.jpg')
        plt.show()
        plt.close()