import os
import unittest
from collections import defaultdict
import logging

from torch.utils.tensorboard import SummaryWriter
from optogenetic_holography import utils
from optogenetic_holography.algorithms import phase_gercherberg_saxton_2D, bin_amp_gercherberg_saxton_2D
from optogenetic_holography.optics import optics_backend as opt
from optogenetic_holography.utils import init_writer

output_path = "./output/"
input_path = "./input/"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

resolution = (225, 225)
z = 10 * opt.cm
wavelength = 488 * opt.nm
pixel_pitch = (10 * opt.um, 10 * opt.um)
radius = 6 * opt.mm
focal_length = 20 * opt.cm
max_iter = 1500

start_wf = opt.Wavefront(wavelength, pixel_pitch, resolution=resolution)
#start_wf = opt.RandomPhaseMask().forward(start_wf)

propagator = opt.FresnelPropagator(z)

img_name = "hexagon.jpg"
target_wf = opt.Wavefront.from_image(input_path + img_name, wavelength, pixel_pitch)


class TestGercherbergSaxton(unittest.TestCase):

    def test_2D_phase_gercherberg_saxton(self):
        writer = init_writer(output_path, "2D_phase_gs")

        #wf_target.plot(intensity=defaultdict(str, title="target_img"))

        holo_wf = phase_gercherberg_saxton_2D(start_wf, target_wf.amplitude, propagator, writer, max_iter=max_iter)
        holo_wf.plot(phase=defaultdict(str, title="gs_phase__holo", path=output_path, save=True))

        recon_amp = propagator.forward(holo_wf)
        recon_amp.plot(intensity=defaultdict(str, title="gs_phase__recon_img", path=output_path, save=True, normalize=True))

        writer.close()

    def test_2D_bin_amp_gercherberg_saxton(self):
        writer = init_writer(output_path, "2D_bin_amp_gs")

        #wf_target.plot(intensity=defaultdict(str, title="target_img"))

        holo_wf = bin_amp_gercherberg_saxton_2D(start_wf, target_wf.amplitude, propagator, writer, max_iter=max_iter)
        holo_wf.plot(intensity=defaultdict(str, title="gs_bin__amp_holo", path=output_path, save=True))

        recon_amp = propagator.forward(holo_wf)
        recon_amp.plot(intensity=defaultdict(str, title="gs_bin__recon_img", path=output_path, save=True, normalize=True))

        writer.close()


