import torch
from torch import square, polar

incident_field = None


def define_common_wavefields(resolution, wavelength, pixel_pitch):
    global incident_field
    incident_field = Wavefield(torch.ones(resolution, dtype=torch.cfloat), wavelength, pixel_pitch)


class Wavefield:
    def __init__(self, u: torch.cfloat, wavelength, pixel_pitch):
        self.u = u
        self.resolution = u.shape
        self.wavelength = wavelength
        self.pixel_pitch = pixel_pitch

    @property
    def amplitude(self):
        return self.u.abs()

    @property
    def phase(self):
        return self.u.angle()

    @property
    def intensity(self):
        return square(self.amplitude)

    @property
    def total_intensity(self):
        return float(self._numpy(self.intensity.sum()))

    def replace_amplitude(self, target_amplitude):
        self.u = polar(target_amplitude, self.phase)

    def _numpy(self, tt):
        return tt.cpu().detach().numpy()

    def assert_equal(self, other_field):
        return torch.allclose(self.u, other_field.u)
