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
wf = opt.Wavefront.from_image(input_path + img_name, wavelength, pixel_pitch)
#random_phase_mask = opt.RandomPhaseMask()
#wf = random_phase_mask.forward(wf)

class TestFourierPropagator(unittest.TestCase):

    def test_forward_backward(self):
        propagator = opt.FourierLensPropagator(radius, focal_length)
        f_field = propagator.forward(wf)
        b_field = propagator.backward(f_field)

        wf.plot(intensity=defaultdict(str, title="start_field"))
        f_field.plot(intensity=defaultdict(str, title="fresnel_propagated"))
        b_field.plot(intensity=defaultdict(str, title="fresnel_back_propagated"))

        self.assertAlmostEqual(wf.total_intensity, f_field.total_intensity, delta=1e-1)
        self.assertTrue(wf.assert_equal(b_field))


class TestFresnelPropagator(unittest.TestCase):

    def test_forward_backward(self):
        propagator = opt.FresnelPropagator(z)
        wf_f = propagator.forward(wf)
        wf_b = propagator.backward(wf_f)

        wf.plot(intensity=defaultdict(str, title="start_field"))
        wf_f.plot(intensity=defaultdict(str, title="fresnel_propagated"))
        wf_b.plot(intensity=defaultdict(str, title="fresnel_back_propagated"))

        self.assertAlmostEqual(wf.total_intensity, wf_f.total_intensity, delta=1e-1)  # fixme failling with random phase
        self.assertTrue(wf.assert_equal(wf_b))