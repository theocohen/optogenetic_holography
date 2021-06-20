import torch
import numpy as np
from optogenetic_holography.optics import optics_backend as opt
import logging

from collections import defaultdict
import unittest

output_path = "./../output/"
input_path = "./../input/"

z = 10 * opt.cm
wavelength = 488 * opt.nm
pixel_pitch = (10 * opt.um, 10 * opt.um)
radius = 6 * opt.mm
focal_length = 20 * opt.cm

img_name = "hexagon.jpg"
wf = opt.Wavefront.from_images(input_path + img_name)
#random_phase_mask = opt.RandomPhaseMask()
#wf = random_phase_mask.forward(wf)

class TestFourierPropagator(unittest.TestCase):

    def test_forward_backward(self):
        propagator = opt.FourierLensPropagator(radius, focal_length)
        f_field = propagator.forward(wf)
        b_field = propagator.backward(f_field)

        wf.plot_old(intensity=defaultdict(str, title="start_field"))
        f_field.plot_old(intensity=defaultdict(str, title="fresnel_propagated"))
        b_field.plot_old(intensity=defaultdict(str, title="fresnel_back_propagated"))

        self.assertAlmostEqual(wf.total_intensity, f_field.total_intensity, delta=1e-1)
        self.assertTrue(wf.assert_equal(b_field))


class TestFresnelPropagator(unittest.TestCase):

    def test_interference(self):
        holo_wf = opt.Wavefront(wavelength, pixel_pitch, resolution=(2,2))
        holo_wf.phase = torch.Tensor([[-np.pi/2, 0], [np.pi/2, np.pi]]).double()
        holo_wf.plot_old(phase=defaultdict(str, title="holo"))

        propagator = opt.FourierLensPropagator(0,0)
        recon_wf = propagator.forward(holo_wf)
        recon_wf.plot_old(intensity=defaultdict(str, title="recon"))

        #holo_wf.amplitude = ((holo_wf.phase > - np.pi / 2) & (holo_wf.phase <= np.pi / 2)).double()
        holo_wf.amplitude = (holo_wf.phase > 0).double()
        holo_wf.plot_old(intensity=defaultdict(str, title="bin_holo"))
        recon_wf = propagator.forward(holo_wf)
        recon_wf.plot_old(intensity=defaultdict(str, title="bin_recon"))


    def test_forward_backward(self):
        propagator = opt.FresnelPropagator(wavelength, pixel_pitch, z)
        wf_f = propagator.forward(wf)
        wf_b = propagator.backward(wf_f)

        wf.plot_old(intensity=defaultdict(str, title="start_field"))
        wf_f.plot_old(intensity=defaultdict(str, title="fresnel_propagated"))
        wf_b.plot_old(intensity=defaultdict(str, title="fresnel_back_propagated"))

        self.assertAlmostEqual(wf.total_intensity, wf_f.total_intensity, delta=1e-1)  # fixme failling with random phase
        self.assertTrue(wf.assert_equal(wf_b))

    def test_forward_3D(self):
        image_folder = "digits/*.jpg"
        z_stack = z + torch.arange(-5, 5, 1) * opt.mm
        wf = opt.Wavefront.from_images(input_path + image_folder, wavelength, pixel_pitch)

        propagator = opt.FresnelPropagator(z_stack)
        wf_f = propagator.forward(wf)
        wf_b = propagator.backward(wf_f)

        wf.plot_old_old(intensity=defaultdict(str, save=True, path=output_path+"digits/", title="start_field"))
        wf_f.plot_old_old(intensity=defaultdict(str, save=True, path=output_path+"digits/", title="fresnel_propagated"))
        wf_b.plot_old_old(intensity=defaultdict(str, save=True, path=output_path+"digits/", title="fresnel_back_propagated"))

        np.testing.assert_array_almost_equal(wf.total_intensity, wf_f.total_intensity, decimal=1)  # fixme failling with random phase
        self.assertTrue(wf.assert_equal(wf_b))

    def test_forward_multi_plane(self):
        z_stack = z + torch.arange(-5, 5, 1) * opt.mm

        propagator = opt.FresnelPropagator(z_stack)
        wf_f = propagator.forward(wf)
        wf_b = propagator.backward(wf_f)

        #wf.plot_old_old(intensity=defaultdict(str, save=True, path=output_path+"digits/", title="start_field"))
        wf_f.plot_old_old(intensity=defaultdict(str, save=True, path=output_path+"multiplane/", title="prop_plane_"))
        wf_b.plot_old_old(intensity=defaultdict(str, save=True, path=output_path+"multiplane/", title="back_prop"))

        #np.testing.assert_array_almost_equal(wf.total_intensity, wf_f.total_intensity, decimal=1)  # fixme failling with random phase
        #self.assertTrue(wf.assert_equal(wf_b))
