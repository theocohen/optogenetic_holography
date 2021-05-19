import unittest
import torch_testing as tt
import torch

import optogenetic_holography.wavefield as wavefield
from optogenetic_holography.utils.image_utils import load_image, save_image

resolution = (1080, 1920)
wavefield.define_common_wavefields(resolution)
output_path = "./output/"
input_path = "./input/"


class TestWavefield(unittest.TestCase):

    def test_incident_field(self):
        tt.assert_equal(wavefield.incident_field.amplitude, torch.ones(resolution))
        tt.assert_equal(wavefield.incident_field.phase, torch.zeros(resolution))
        tt.assert_equal(wavefield.incident_field.intensity, torch.ones(resolution))
        self.assertEqual(wavefield.incident_field.total_intensity, resolution[0] * resolution[1])

    def test_image_field(self):
        # fixme weird plt when setting (vmin, vmax) = (0,1)
        img_name = "square"
        field = wavefield.Wavefield(load_image(input_path, img_name + ".jpg"))
        save_image(field, output_path, img_name)