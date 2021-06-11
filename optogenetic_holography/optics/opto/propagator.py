import numpy as np
import torch

from optogenetic_holography.optics.propagator import Propagator
from optogenetic_holography.optics.opto.wavefront import Wavefront



class FourierFresnelPropagator(Propagator):
    #Assumes propagates through lens, from a distance d=f to lens and to focal length from lens.
    #   Then, we use Fresnel propagation for the remaining short distance z

    def __init__(self, radius, focal_length, wavelength, pixel_pitch, z):
        self.fourier_lens_propagator = FourierLensPropagator(radius, focal_length)
        self.fresnel_propagator = FresnelPropagator(wavelength, pixel_pitch, z)

    def forward(self, wf):
        return self.fresnel_propagator.forward(
               self.fourier_lens_propagator.forward(wf))

    def backward(self, wf):
        return self.fourier_lens_propagator.backward(
               self.fresnel_propagator.backward(wf))



class FourierLensPropagator(Propagator):
    """ FIXME """

    def __init__(self, radius, focal_length):
        pass

    def forward(self, wf) -> Wavefront:
        propagated_wf = wf.copy()
        propagated_wf.u = torch.fft.fftshift(torch.fft.fft2(wf.u, norm="ortho"))
        return propagated_wf

    def backward(self, wf) -> Wavefront:
        propagated_wf = wf.copy()
        propagated_wf.u = torch.fft.ifft2(torch.fft.ifftshift(wf.u), norm="ortho")
        return propagated_wf

class FresnelPropagator(Propagator):

    def __init__(self, wavelength, pixel_pitch, z):
        self.wavelength = wavelength
        self.pixel_pitch = pixel_pitch
        z = torch.tensor([z]) if not torch.is_tensor(z) else z
        self.z = z.reshape((1, -1, 1, 1)).float()
        self.precomputed_H = None

    def forward(self, wf) -> Wavefront:
        if self.precomputed_H is None:
            k = 2 * np.pi / self.wavelength

            nx, ny = wf.resolution
            dx, dy = self.pixel_pitch

            delta_x = 1 / (nx * dx)
            delta_y = 1 / (ny * dy)

            f_x = torch.arange(-nx / 2 + 1, nx / 2 + 1, 1, dtype=torch.float64) * delta_x
            f_y = torch.arange(-ny / 2 + 1, ny / 2 + 1, 1, dtype=torch.float64) * delta_y
            f_y, f_x = torch.meshgrid(f_x, f_y)

            H_exp = k - np.pi * self.wavelength * (f_x ** 2 + f_y ** 2)
            self.precomputed_H = torch.exp(1j * H_exp.float() * self.z).to(wf.device)

        propagated_wf = wf.copy()
        propagated_wf.depth = self.z.shape[1]
        G = torch.fft.fftshift(torch.fft.fft2(wf.u, norm='ortho'))
        propagated_wf.u = torch.fft.ifft2(torch.fft.ifftshift(G * self.precomputed_H), norm='ortho')
        return propagated_wf

    def backward(self, wf) -> Wavefront:
        propagated_wf = wf.copy()
        G = torch.fft.fftshift(torch.fft.fft2(wf.u, norm='ortho'))
        propagated_wf.u = (torch.fft.ifft2(torch.fft.ifftshift(G / self.precomputed_H), norm='ortho')) # inverse kernel
        return propagated_wf

class RandomPhaseMask(Propagator):

    def forward(self, wf) -> Wavefront:
        masked_wf = wf.copy()
        masked_wf.amplitude = wf.amplitude
        masked_wf.phase = np.pi * (1 - 2 * torch.rand(wf.resolution)).float()  # between -pi to pi
        return masked_wf

    def backward(self, wf) -> Wavefront:
        pass
