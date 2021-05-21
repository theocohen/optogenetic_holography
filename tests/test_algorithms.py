import os
import unittest
from collections import defaultdict
import logging

from torch.utils.tensorboard import SummaryWriter
from optogenetic_holography import utils
from optogenetic_holography.algorithms import gercherberg_saxton
from optogenetic_holography.optics import optics_backend as opt

output_path = "./output/"
input_path = "./input/"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

summaries_dir = os.path.join(output_path + 'summaries')
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(summaries_dir)

resolution = (225, 225)
z = 10 * opt.cm
wavelength = 488 * opt.nm
pixel_pitch = (10 * opt.um, 10 * opt.um)
radius = 6 * opt.mm
focal_length = 20 * opt.cm
max_iter = 1500

wf_source = opt.Wavefront(wavelength, pixel_pitch, resolution=resolution)
wf = opt.RandomPhaseMask().forward(wf_source)

propagator = opt.FresnelPropagator(z)

img_name = "hexagon.jpg"
wf_target = opt.Wavefront.from_image(input_path + img_name, wavelength, pixel_pitch)


class TestGercherbergSaxton(unittest.TestCase):

    def test_2D_phase_gercherberg_saxton(self):
        wf_target.plot(intensity=defaultdict(str, title="target_img"))

        wf_holo = gercherberg_saxton(wf_source, wf_target.amplitude, propagator, writer, max_iter=max_iter)
        wf_holo.plot(phase=defaultdict(str, title="gs_phase_holo", path=output_path, save=True))

        wf_object = propagator.forward(wf_holo)
        wf_object.plot(intensity=defaultdict(str, title="reconstructed_img", path=output_path, save=True, normalize=True))

        writer.close()


