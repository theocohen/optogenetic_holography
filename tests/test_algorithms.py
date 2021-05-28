import unittest
from collections import defaultdict
import logging

from torch import nn

from optogenetic_holography.algorithms import phase_gercherberg_saxton_2D, bin_amp_gercherberg_saxton_2D, phase_sgd_2D, \
    bin_amp_phase_sgd_2D, bin_amp_phase_sgd_2D, bin_amp_amp_sgd_2D
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
max_iter = 5000
lr = 0.1
scale_loss = False

start_wf = opt.Wavefront(wavelength, pixel_pitch, resolution=resolution)
start_wf = opt.RandomPhaseMask().forward(start_wf)

propagator = opt.FresnelPropagator(z)

img_name = "hexagon.jpg"
target_wf = opt.Wavefront.from_image(input_path + img_name, wavelength, pixel_pitch)


def bin_amp_modulation(holo_wf, method="phase"):
    if method == "amplitude":
        return (holo_wf.amplitude > holo_wf.amplitude.mean()).float()
    elif method == "phase":
        return (holo_wf.phase > 0).float()


class TestGercherbergSaxton(unittest.TestCase):

    def test_phase_gercherberg_saxton_2D(self):
        experiment = "phase_GS_2D"
        writer = init_writer(output_path, experiment)

        #wf_target.plot(intensity=defaultdict(str, title="target_img"))

        holo_wf = phase_gercherberg_saxton_2D(start_wf, target_wf.amplitude, propagator, writer, max_iter=max_iter, scale_loss=scale_loss)
        holo_wf.plot(phase=defaultdict(str, title=experiment+"__holo", path=output_path, save=True))

        recon_amp = propagator.forward(holo_wf)
        recon_amp.plot(intensity=defaultdict(str, title=experiment+"__recon", path=output_path, save=True, normalize=True))

        writer.close()

    def test_bin_amp_gercherberg_saxton_2D(self):
        experiment = "bin_amp_GS_2D"
        writer = init_writer(output_path, experiment)

        #wf_target.plot(intensity=defaultdict(str, title="target_img"))

        holo_wf = bin_amp_gercherberg_saxton_2D(start_wf, target_wf.amplitude, propagator, bin_amp_modulation, writer, max_iter=max_iter, scale_loss=scale_loss)
        holo_wf.plot(intensity=defaultdict(str, title=experiment+"__holo", path=output_path, save=True))

        recon_amp = propagator.forward(holo_wf)
        recon_amp.plot(intensity=defaultdict(str, title=experiment+"__recon", path=output_path, save=True, normalize=True))

        writer.close()

    def test_phase_sgd_2D(self):
        experiment = "phase_SGD_2D"
        loss_fn = nn.MSELoss()

        writer = init_writer(output_path, experiment)

        #wf_target.plot(intensity=defaultdict(str, title="target_img"))

        holo_wf = phase_sgd_2D(start_wf, target_wf.amplitude, propagator, loss_fn, writer, max_iter=max_iter, lr=lr, scale_loss=scale_loss)
        holo_wf.plot(phase=defaultdict(str, title=experiment + "__holo", path=output_path, save=True))

        recon_amp = propagator.forward(holo_wf)
        recon_amp.plot(intensity=defaultdict(str, title=experiment + "__recon", path=output_path, save=True, normalize=True))

        writer.close()

    def test_bin_amp_phase_sgd_2D(self):
        experiment = "bin_amp_phase-SGD_2D"
        loss_fn = nn.MSELoss()

        writer = init_writer(output_path, experiment)

        #wf_target.plot(intensity=defaultdict(str, title="target_img"))

        holo_wf = bin_amp_phase_sgd_2D(start_wf, target_wf.amplitude, propagator, loss_fn, bin_amp_modulation, writer, max_iter=max_iter, lr=lr, scale_loss=scale_loss)
        holo_wf.plot(phase=defaultdict(str, title=experiment + "__holo", path=output_path, save=True))

        recon_amp = propagator.forward(holo_wf)
        recon_amp.plot(intensity=defaultdict(str, title=experiment + "__recon", path=output_path, save=True, normalize=True))

        writer.close()

    def test_bin_amp_amp_sgd_2D(self):
        experiment = "bin_amp_amp-SGD_2D"
        loss_fn = nn.MSELoss()

        writer = init_writer(output_path, experiment)

        #wf_target.plot(intensity=defaultdict(str, title="target_img"))

        holo_wf = bin_amp_amp_sgd_2D(start_wf, target_wf.amplitude, propagator, loss_fn, bin_amp_modulation, writer, max_iter=max_iter, lr=lr, scale_loss=scale_loss)
        holo_wf.plot(intensity=defaultdict(str, title=experiment + "__holo", path=output_path, save=True))

        recon_amp = propagator.forward(holo_wf)
        recon_amp.plot(intensity=defaultdict(str, title=experiment + "__recon", path=output_path, save=True, normalize=True))

        writer.close()