import unittest
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
import pyoptica as po
import diffractsim as difs

from optogenetic_holography.propagator import FresnelPropagator
import optogenetic_holography.wavefield as wavefield
from optogenetic_holography.utils.image_utils import load_image, save_image

output_path = "./output/"
input_path = "./input/"

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
z = 20 * cm
wavelength = 638 * nm
pixel_pitch = (10 * um, 10 * um)


class TestAllPropagator(unittest.TestCase):

    def test_fresnel_propagator(self):
        img_name = "square"
        start_field = wavefield.Wavefield(load_image(input_path, img_name + ".jpg"), wavelength, pixel_pitch)

        # This project
        wf_opto = self.propagate_opto(start_field)

        # PyOptica
        wf_pyoptica = self.propagate_pyoptica()

        # Diffractsim
        wf_difs = self.propagate_difs(start_field.resolution, img_name)

        # Chris' code
        wf_chris = self.propagate_chris(start_field._numpy(start_field.u), wavelength, z, pixel_pitch)

    def propagate_opto(self, start_field):
        propagator = FresnelPropagator()
        wf_opto = propagator.forward(start_field, z)
        save_image(wf_opto, output_path, "OPTO -  propagated", plot=True)

        return wf_opto

    def propagate_pyoptica(self):
        wvl = 638 * u.nm  # wavelength
        npix = 400  # Nx
        pixel_scale = 10 * u.um  # dx
        f = 20 * u.cm  # z
        w = 80 * pixel_scale
        h = 80 * pixel_scale
        wf = po.Wavefront(wvl, pixel_scale, npix)
        ap = po.RectangularAperture(w, h)
        wf = wf * ap
        fs_f = po.FreeSpace(f)
        wf_forward = wf * fs_f
        #wf_forward.plot(intensity=dict(title='PYOPTICA -  propagated', vmin=0, vmax=1, cmap='gray'))
        wf_forward.plot(intensity=dict(title='PYOPTICA -  propagated', cmap='gray'))
        plt.show()

        return wf_forward

    def propagate_difs(self, resolution, img_name):
        Nx, Ny = resolution
        difs.set_backend("CPU")  # Change the string to "CUDA" to use GPU acceleration
        extent = (Nx * pixel_pitch[0], Ny * pixel_pitch[1])
        F = difs.MonochromaticField(wavelength=wavelength, extent_x=extent[0], extent_y=extent[1], Nx=Nx, Ny=Ny)
        F.add_aperture_from_image(input_path + img_name + '.jpg')
        #F.add_rectangular_slit(0, 0, 80 * pixel_pitch[0], 80 * pixel_pitch[1])  # alternative
        rgb = F.compute_colors_at(z)
        F.plot(rgb)

        return rgb

    def propagate_chris(self, wave_field_start, wavelength, z, sampling_rates):
        k = 2 * np.pi / wavelength  # angular wave number

        nx, ny = wave_field_start.shape  # shape of start 2D wavefront
        dx, dy = sampling_rates

        delta_x = 1 / (nx * dx)
        delta_y = 1 / (ny * dy)

        u = np.arange(-nx / 2 + 1, nx / 2 + 1, 1) * delta_x
        v = np.arange(-ny / 2 + 1, ny / 2 + 1, 1) * delta_y
        u, v = np.meshgrid(u, v)

        G = np.fft.fftshift(np.fft.fft2(wave_field_start))  # why fftshift?

        H = np.exp(1j * k * z - np.pi * 1j * wavelength * z * (u ** 2 + v ** 2))  # fourier transformed Fresnel convolution kernel.

        wave_field_end = np.fft.ifft2(np.fft.ifftshift(G * H))  # ifftshift not necessary here
        wave_field_end = np.abs(wave_field_end) ** 2  # get intensity

        # plot result
        plt.imshow(wave_field_end, cmap='gray')
        plt.title('CHRIS - Propagated wave field'), plt.xticks([]), plt.yticks([])
        plt.show()

        return wave_field_end



