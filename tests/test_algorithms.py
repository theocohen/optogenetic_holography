import unittest
from collections import defaultdict
import logging

import torch
from torch import nn

from optogenetic_holography.algorithms import bin_amp_phase_gercherberg_saxton, bin_amp_amp_gercherberg_saxton, \
    phase_sgd, \
    bin_amp_phase_sgd, bin_amp_amp_sgd, bin_amp_amp_sig_sgd
from optogenetic_holography.binarization import from_amp_to_bin_amp, from_phase_to_bin_amp
from optogenetic_holography.optics import optics_backend as opt
from optogenetic_holography.utils import init_writer

output_path = "./output/"
input_path = "./input/"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

MULTIPLANE = False

wavelength = 488 * opt.nm
pixel_pitch = (10 * opt.um, 10 * opt.um)
#radius = 6 * opt.mm
#focal_length = 20 * opt.cm

max_iter = 400
lr = 0.1
scale_loss = False
loss_fn = nn.MSELoss()

if MULTIPLANE: # 3D
    target_wf = opt.Wavefront.from_images(input_path + "digits/*.jpg", wavelength, pixel_pitch, optimize_resolution=False)
    z = 10 * opt.cm + torch.arange(-5, 5, 1) * opt.mm
else:
    # 2D
    target_wf = opt.Wavefront.from_images(input_path + "bahtinov_mask.jpg", wavelength, pixel_pitch, optimize_resolution=False)
    z = 10 * opt.cm

# wf_target.plot(intensity=defaultdict(str, title="target_img"))

start_wf = opt.Wavefront(wavelength, pixel_pitch, target_wf.resolution)

propagator = opt.FresnelPropagator(z)
#propagator = opt.FourierLensPropagator(0,0)
#propagator = opt.FourierFresnelPropagator(0, 0, z)
suppress_center = False


class TestGercherbergSaxton(unittest.TestCase):

    def test_bin_amp_phase_gercherberg_saxton(self):
        experiment = "bin_amp_phase_GS_3D" if MULTIPLANE else "bin_amp_phase_GS_2D"
        setup = 'fresnel-end_bin'

        writer = init_writer(output_path, experiment, setup=setup)

        holo_wf = bin_amp_phase_gercherberg_saxton(start_wf, target_wf.amplitude, propagator, writer, max_iter=max_iter, scale_loss=scale_loss)
        holo_wf.plot(intensity=defaultdict(str, title=experiment+"__holo", path=output_path, save=True))

        recon_wf = propagator.forward(holo_wf)
        recon_wf.plot(intensity=defaultdict(str, title=experiment+"__recon", path=output_path, save=True, suppress_center=suppress_center, normalize=True))

        writer.close()

    def test_bin_amp_amp_gercherberg_saxton(self):
        experiment = "bin_amp_amp_GS_3D" if MULTIPLANE else "bin_amp_amp_GS_2D_"
        setup = 'fourier_fresnel-end_bin-mask'

        # modes = ["otsu", 'yen', 'isodata', 'li', 'minimum', 'mean', 'niblack','sauvola','triangle']
        bin_amp_mode = "otsu"

        writer = init_writer(output_path, experiment + bin_amp_mode, setup)

        diffused_start_wf = opt.RandomPhaseMask().forward(start_wf)
        holo_wf = bin_amp_amp_gercherberg_saxton(diffused_start_wf, target_wf.amplitude, propagator, writer, max_iter=max_iter, scale_loss=scale_loss, bin_amp_mode=bin_amp_mode)
        holo_wf.plot(intensity=defaultdict(str, title=experiment+"__holo", path=output_path, save=True))

        recon_wf = propagator.forward(holo_wf)
        recon_wf.plot(intensity=defaultdict(str, title=experiment+"__recon", path=output_path, save=True, normalize=True, suppress_center=suppress_center))

        writer.close()

    def test_phase_sgd(self):
        experiment = "phase_SGD_3D" if MULTIPLANE else "phase_SGD_2D"

        writer = init_writer(output_path, experiment)

        holo_wf = phase_sgd(start_wf, target_wf.amplitude, propagator, loss_fn, writer, max_iter=max_iter, lr=lr, scale_loss=scale_loss)
        holo_wf.plot(phase=defaultdict(str, title=experiment + "__holo", path=output_path, save=True))

        recon_wf = propagator.forward(holo_wf)
        recon_wf.plot(intensity=defaultdict(str, title=experiment + "__recon", path=output_path, save=True, normalize=True))

        writer.close()

    def test_bin_amp_phase_sgd(self):
        experiment = "bin_amp_phase-SGD_3D" if MULTIPLANE else "bin_amp_phase-SGD_2D"
        setup='fresnel-end_bin'

        writer = init_writer(output_path, experiment, setup=setup)

        holo_wf = bin_amp_phase_sgd(start_wf, target_wf.amplitude, propagator, loss_fn, writer, max_iter=max_iter, lr=lr, scale_loss=scale_loss)
        holo_wf.plot(phase=defaultdict(str, title=experiment + "__holo", path=output_path, save=True))

        recon_wf = propagator.forward(holo_wf)
        recon_wf.plot(intensity=defaultdict(str, title=experiment + "__recon", path=output_path, save=True, normalize=True))

        writer.close()

    """todo def bin_amp_phase_sig_sgd(self)"""

    def test_bin_amp_amp_sgd(self):
        experiment = "bin_amp_amp-SGD_3D" if MULTIPLANE else "bin_amp_amp-SGD_2D_"
        setup = 'fresnel-end_bin-mask'

        #modes = ["otsu", 'yen', 'isodata', 'li', 'minimum', 'mean', 'niblack','sauvola','triangle']
        bin_amp_mode = "otsu"

        writer = init_writer(output_path, experiment + bin_amp_mode, setup=setup)

        diffused_start_wf = opt.RandomPhaseMask().forward(start_wf)
        holo_wf = bin_amp_amp_sgd(diffused_start_wf, target_wf.amplitude, propagator, loss_fn, writer, max_iter=max_iter, lr=lr, scale_loss=scale_loss, bin_amp_mode=bin_amp_mode)
        holo_wf.plot(intensity=defaultdict(str, title=experiment + "__holo", path=output_path, save=True))

        recon_wf = propagator.forward(holo_wf)
        recon_wf.plot(intensity=defaultdict(str, title=experiment + "__recon", path=output_path, save=True, normalize=True))

        writer.close()

    def test_bin_amp_amp_sig_sgd(self):
        experiment = "bin_amp_amp_sig_SGD_3D" if MULTIPLANE else "bin_amp_amp_sig_SGD_2D"

        writer = init_writer(output_path, experiment)

        diffused_start_wf = opt.RandomPhaseMask().forward(start_wf)
        holo_wf = bin_amp_amp_sig_sgd(diffused_start_wf, target_wf.amplitude, propagator, loss_fn, from_amp_to_bin_amp, writer, max_iter=max_iter, lr=lr, scale_loss=scale_loss)
        holo_wf.plot(intensity=defaultdict(str, title=experiment + "__holo", path=output_path, save=True))

        recon_wf = propagator.forward(holo_wf)
        recon_wf.plot(intensity=defaultdict(str, title=experiment + "__recon", path=output_path, save=True, normalize=True))

        writer.close()