from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.fft import fft2, fftshift, ifft2, ifftshift

from optogenetic_holography.wavefield import Wavefield


class Propagator(ABC):

    @abstractmethod
    def forward(self, field: Wavefield) -> Wavefield:
        pass

    @abstractmethod
    def backward(self, field: Wavefield) -> Wavefield:
        pass


class FourierLensPropagator(Propagator):

    def forward(self, field):
        return Wavefield(fftshift(fft2(field.u, norm="ortho")), field.wavelength, field.pixel_pitch)

    def backward(self, field):
        return Wavefield(ifft2(ifftshift(field.u), norm="ortho"), field.wavelength, field.pixel_pitch)


class FresnelPropagator(Propagator):
    def __init__(self):
        self.precomputed_H_exp = None  # assuming fixed wavelength, resolution and pixel_pitch
        self.precomputed_H = {}

    def forward(self, field, z) :
        if z not in self.precomputed_H:

            if self.precomputed_H_exp is None:
                k = 2 * np.pi / field.wavelength

                nx, ny = field.resolution
                dx, dy = field.pixel_pitch

                delta_x = 1 / (nx * dx)
                delta_y = 1 / (ny * dy)

                f_x = torch.arange(-nx / 2 + 1, nx / 2 + 1, 1, dtype=torch.float64) * delta_x
                f_y = torch.arange(-ny / 2 + 1, ny / 2 + 1, 1, dtype=torch.float64) * delta_y
                f_y, f_x = torch.meshgrid(f_x, f_y)

                H_exp = k - np.pi * field.wavelength * (f_x ** 2 + f_y ** 2)
                self.precomputed_H_exp = H_exp

            else:
                H_exp = self.precomputed_H_exp

            H = torch.exp(1j * H_exp * z)
            self.precomputed_H[z] = H_exp
        else:
            H = self.precomputed_H[z]

        G = fftshift(fft2(field.u, norm='ortho'))
        u_out = ifft2(ifftshift(G * H), norm='ortho')
        return Wavefield(u_out, field.wavelength, field.pixel_pitch)

    def backward(self, field, z):
        return self.forward(field, -z)
