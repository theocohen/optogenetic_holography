import cv2
import pyoptica as po
import matplotlib.pyplot as plt

#from optogenetic_holography.optics.wavefront import WavefrontInterface


class Wavefront():
    def __init__(self, wavelength, pixel_pitch, resolution):
        """assumes square shape"""
        self.wf = po.Wavefront(wavelength, pixel_pitch[0], resolution[0])

    @classmethod
    def from_image(cls, path, wavelength, pixel_pitch, scale_intensity=1):
        img = cv2.imread(path, 0)
        wf = Wavefront(wavelength, pixel_pitch, img.shape)
        wf.amplitude = img / img.max() * scale_intensity
        return wf

    @property
    def wavelength(self):
        return self.wf.wavelength

    @property
    def pixel_pitch(self):
        return (self.wf.pixel_scale, self.wf.pixel_scale)

    @property
    def resolution(self):
        return self.wf.npix, self.wf.npix

    @property
    def amplitude(self):
        return self.wf.amplitude

    @amplitude.setter
    def amplitude(self, new_amplitude):
        self.wf.amplitude = new_amplitude

    @property
    def phase(self):
        return self.wf.phase

    @phase.setter
    def phase(self, new_phase):
        self.wf.amplitude = new_phase

    @property
    def intensity(self):
        return self.wf.intensity

    @property
    def total_intensity(self):
        return self.intensity.sum()

    def replace_amplitude(self, target_amplitude):
        self.wf.amplitude = target_amplitude

    def assert_equal(self, wf_2):
        return self.wf.__eq__(wf_2.wf)

    def plot(self, fig_options=None, **kwargs):
        self.wf.plot(fig_options=fig_options, **kwargs)
        plt.show()

    def copy(self):
        return Wavefront(self.wavelength, self.pixel_pitch, self.resolution)