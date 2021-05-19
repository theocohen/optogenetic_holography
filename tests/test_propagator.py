import unittest
import torch

from optogenetic_holography.propagator import FourierLensPropagator, FresnelPropagator
import optogenetic_holography.wavefield as wavefield
from optogenetic_holography.utils.image_utils import load_image, save_image

output_path = "./output/"
input_path = "./input/"

resolution = (1080, 1920)
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
z = 20 * cm
wavelength = 638 * nm
pixel_pitch = (10 * um, 10 * um)
wavefield.define_common_wavefields(resolution, wavelength, pixel_pitch)


class TestFourierPropagator(unittest.TestCase):

    def test_fourier_lens_propagator(self):
        propagator = FourierLensPropagator()
        f_field = propagator.forward(wavefield.incident_field)
        b_field = propagator.backward(f_field)

        self.assertEqual(f_field.total_intensity, wavefield.incident_field.total_intensity)
        self.assertTrue(b_field.assert_equal(wavefield.incident_field))


class TestFresnelPropagator(unittest.TestCase):

    def test_simple_fresnel(self):
        wavelength = 1
        z = 40
        pixel_pitch = (1,1)
        u = torch.tensor([[0, 0, 0, 0],[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=torch.cfloat)
        simple_field = wavefield.Wavefield(u, wavelength, pixel_pitch)
        propagator = FresnelPropagator()
        f_field = propagator.forward(simple_field, z)
        b_field = propagator.backward(f_field, z)

        save_image(simple_field, output_path, "simple_field")
        save_image(f_field, output_path, "simple_field_f")

    def test_fresnel_propagator(self):
        img_name = "square"
        start_field = wavefield.Wavefield(load_image(input_path, img_name + ".jpg"), wavelength, pixel_pitch)

        propagator = FresnelPropagator()
        f_field = propagator.forward(start_field, z)
        b_field = propagator.backward(f_field, z)

        save_image(start_field, output_path, "start_field")
        save_image(f_field, output_path, "fresnel_propagated")
        save_image(b_field, output_path, "fresnel_propagated return")

        self.assertAlmostEqual(start_field.total_intensity, f_field.total_intensity, delta=1e-1)
        self.assertTrue(start_field.assert_equal(b_field))