import numpy as np
import torch
from torch.fft import fft2, fftshift, ifft2, ifftshift

from optogenetic_holography.optics.propagator import Propagator
from optogenetic_holography.optics.opto.wavefront import Wavefront


"""
class FourierFresnelPropagator(Propagator):
    #Assumes propagates through lens, from a distance d=f to lens and to focal length from lens.
    #   Then, we use Fresnel propagation for the remaining short distance z

    def __init__(self):
        self.fourier_lens_propagator = FourierLensPropagator()
        self.fresnel_propagator = FresnelPropagator()

    def forward(self, field, z):
        return self.fresnel_propagator.forward(
               self.fourier_lens_propagator.forward(field), z)

    def backward(self, field, z):
        return self.fresnel_propagator.backward(
               self.fourier_lens_propagator.backward(field), z)
"""


class FourierLensPropagator(Propagator):

    def __init__(self, radius, focal_length):
        pass

    def forward(self, wf):
        propagated_wf = Wavefront(wf.wavelength, wf.pixel_pitch, wf.resolution)
        propagated_wf.u = fftshift(fft2(wf.u, norm="ortho"))
        return propagated_wf

    def backward(self, wf):
        propagated_wf = Wavefront(wf.wavelength, wf.pixel_pitch, wf.resolution)
        propagated_wf.u = ifft2(ifftshift(wf.u), norm="ortho")
        return propagated_wf

class FresnelPropagator(Propagator):
    def __init__(self):
        self.precomputed_H_exp = None  # assuming fixed wavelength, resolution and pixel_pitch
        self.precomputed_H = {}

    def forward(self, wf, z) :
        if z not in self.precomputed_H:

            if self.precomputed_H_exp is None:
                k = 2 * np.pi / wf.wavelength

                nx, ny = wf.resolution
                dx, dy = wf.pixel_pitch

                delta_x = 1 / (nx * dx)
                delta_y = 1 / (ny * dy)

                f_x = torch.arange(-nx / 2 + 1, nx / 2 + 1, 1, dtype=torch.float64) * delta_x
                f_y = torch.arange(-ny / 2 + 1, ny / 2 + 1, 1, dtype=torch.float64) * delta_y
                f_y, f_x = torch.meshgrid(f_x, f_y)

                H_exp = k - np.pi * wf.wavelength * (f_x ** 2 + f_y ** 2)
                self.precomputed_H_exp = H_exp

            else:
                H_exp = self.precomputed_H_exp

            H = torch.exp(1j * H_exp * z)
            self.precomputed_H[z] = H_exp
        else:
            H = self.precomputed_H[z]

        propagated_wf = Wavefront(wf.wavelength, wf.pixel_pitch, wf.resolution)
        G = fftshift(fft2(wf.u, norm='ortho'))
        propagated_wf.u = ifft2(ifftshift(G * H), norm='ortho')
        return propagated_wf

    def backward(self, field, z):
        return self.forward(field, -z)

class RandomPhaseMask(Propagator):

    def forward(self, wf):
        masked_wf = Wavefront(wf.wavelength, wf.pixel_pitch, wf.resolution)
        masked_wf.amplitude = wf.amplitude
        masked_wf.phase = np.pi * (1 - 2 * torch.rand(wf.resolution))  # between -pi to pi
        return masked_wf

    def backward(self, wf):
        pass
