import unittest
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import cv2

import astropy.units as u
import pyoptica as po
import diffractsim as difs

from optogenetic_holography.optics import optics_backend as opt

difs.set_backend("CPU")  # Change the string to "CUDA" to use GPU acceleration

output_path = "../output/"
input_path = "../input/"

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

img_name = "square"
wavelength = 488
npix = 400
pixel_scale = 4
f = 20
w = 80
h = 80
z = 0
r = 6


def pytopica_load_image(path, wavelength, pixel_scale):
    img = cv2.imread(path, 0)
    wf = po.Wavefront(wavelength, pixel_scale, img.shape[0])
    wf.amplitude = img / img.max()
    return wf


class TestAllPropagator(unittest.TestCase):

    def test_fresnel_propagator(self):

        wf_pyoptica = self.propagate_pyoptica(img_name, wavelength, pixel_scale, z)
        wf_difs = self.propagate_difs(img_name, wavelength, pixel_scale, z)
        wf_opto = self.propagate_opto(img_name, wavelength, pixel_scale, z)

        #wf_chris = self.propagate_chris(start_field._numpy(start_field.u), wavelength, z, pixel_pitch)

    def propagate_opto(self, img_name, wavelength, pixel_scale, z):
        wavelength = wavelength * nm
        pixel_scale = (pixel_scale * um, pixel_scale * um)
        z = z * cm

        square_field = opt.Wavefront.from_image(input_path + img_name + '.jpg', wavelength, pixel_scale)

        free_space = opt.FresnelPropagator(z)
        f_end = free_space.forward(square_field)

        f_end.plot(intensity=defaultdict(str, title="OPTO"))

        return f_end

    def propagate_pyoptica(self, img_name, wavelength, pixel_scale, z):
        wavelength = wavelength * u.nm
        pixel_scale = pixel_scale * u.um
        z = z * u.cm

        wf = pytopica_load_image(input_path + img_name + '.jpg', wavelength, pixel_scale)
        #wf.plot(intensity=dict(title='PYOPTICA - Input', cmap='gray'))
        #plt.show()

        fs = po.FreeSpace(z)

        wf_f = wf * fs

        wf_f.plot(intensity=dict(title='PYOPTICA -  2f system', cmap='gray'))
        plt.show()

        return wf_f

    def propagate_difs(self, img_name, wavelength, pixel_scale, z):
        wavelength = wavelength * nm
        pixel_scale = pixel_scale * um
        z = z * cm
        extent = npix * pixel_scale

        F = difs.MonochromaticField(wavelength=wavelength, extent_x=extent, extent_y=extent, Nx=npix, Ny=npix, intensity=1)
        F.add_aperture_from_image(input_path + img_name + '.jpg')
        F.propagate(z)

        rgb = F.get_colors()
        F.plot(rgb, xlim=[-1, 1], ylim=[-1, 1])

        return rgb

    def propagate_chris(self, wave_field_start, wavelength, z, sampling_rates):
        """technically the same code as this project"""
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


class TestLens(unittest.TestCase):

    def test_square(self):

        wf_pyoptica = self.propagate_pyoptica(img_name, wavelength, npix, pixel_scale, w, h, f, r, z)
        wf_difs = self.propagate_difs(img_name, wavelength, npix, pixel_scale, w, h, f, r, z)
        wf_opto = self.propagate_opto(img_name, wavelength, pixel_scale, z)


    def propagate_opto(self, img_name, wavelength, pixel_scale, z):
        wavelength = wavelength * nm
        pixel_scale = (pixel_scale * um, pixel_scale * um)
        z = z * cm

        square_field = opt.Wavefront.from_image(input_path + img_name + '.jpg', wavelength, pixel_scale)

        lens = opt.FourierLensPropagator(0,0)
        f_focal = lens.forward(square_field)

        free_space = opt.FresnelPropagator(z)
        f_end = free_space.forward(f_focal)

        f_end.plot(intensity=defaultdict(str, title="OPTO"))

        return f_end

    def propagate_pyoptica(self, img_name, wavelength, npix, pixel_scale, w, h, f, r, z):
        wavelength = wavelength * u.nm
        pixel_scale = pixel_scale * u.um
        f = f * u.cm
        z = z * u.cm
        w = w * pixel_scale
        h = h * pixel_scale
        r = r * u.mm

        #wf = po.Wavefront(wavelength, pixel_scale, npix)
        #ap = po.RectangularAperture(w, h)
        #wf = wf * ap

        wf = pytopica_load_image(input_path + img_name + '.jpg', wavelength, pixel_scale)
        #wf.plot(intensity=dict(title='PYOPTICA - Input', cmap='gray'))
        #plt.show()

        fs_f = po.FreeSpace(f)
        fs_b = po.FreeSpace(-f)
        fs_small = po.FreeSpace(z)
        lens = po.ThinLens(2 * r, f)

        wf_f1 = wf * fs_f
        wf_lens = wf_f1 * lens
        wf_f2 = wf_lens * fs_f
        #wf_f2 = wf_f2 * fs_b
        #wf_lens = wf_f2 * lens
        #wf_bis = wf_lens * fs_b

        wf_f3 = wf_f2 * fs_small

        wf_f3.plot(intensity=dict(title='PYOPTICA -  2f system', cmap='gray'))
        #wf_bis.plot(intensity=dict(title='PYOPTICA -  2f system', cmap='gray'))
        plt.show()

        return wf_f3

    def propagate_difs(self, img_name, wavelength, npix, pixel_scale, w, h, f, r, z):
        wavelength = wavelength * nm
        pixel_scale = pixel_scale * um
        f = f * cm
        z = z * cm
        w = w * pixel_scale
        h = h * pixel_scale
        r = r * mm
        extent = npix * pixel_scale

        F = difs.MonochromaticField(wavelength=wavelength, extent_x=extent, extent_y=extent, Nx=npix, Ny=npix, intensity=1)

        F.add_aperture_from_image(input_path + img_name + '.jpg')
        #F.add_rectangular_slit(0, 0, w, h)

        F.propagate(f)
        F.add_lens(f=f)
        F.add_circular_slit(0, 0, r)
        F.propagate(f)
        F.propagate(z)

        rgb = F.get_colors()
        F.plot(rgb, xlim=[-3, 3], ylim=[-3, 3])

        return rgb





