import glob
import os
import logging
#from pytorch_ssim import ssim
import sys
from datetime import datetime

import cv2
import numpy as np
import torch

from piq import psnr, ssim
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter

from optogenetic_holography.optics import optics_backend as opt


def config_logger(summary_dir, run_dir):
    file_handler = logging.FileHandler(summary_dir + '/logs.log')
    console_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(format='%(asctime)s - %(run_dir)s - %(message)s', level=logging.INFO,
                        handlers=[file_handler, console_handler])
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.run_dir = run_dir
        return record

    logging.setLogRecordFactory(record_factory)

def mkdir(path):
    if os.path.exists(path):
        logging.info(f"Deleting summaries at {path}")
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))
        os.removedirs(path)
    os.makedirs(path)


def load_mask(mask_path, device):
    if mask_path == '':
        return None
    images = np.stack([cv2.imread(file, 0) for file in sorted(glob.glob(mask_path))]) / 255.0
    return torch.tensor(images, dtype=torch.float32).reshape((1,) + images.shape).to(device)


def init_writer(output_path, experiment, setup=None):
    summary_dir = os.path.join(output_path, experiment)
    summary_dir = os.path.join(summary_dir, setup if (setup is not None and setup is not '') else datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    mkdir(summary_dir)
    return SummaryWriter(summary_dir), summary_dir


def assert_phase_unchanged(amplitude, holo_wf, start_wf, just_check_first=True):
    if just_check_first:
        phase_shift = (holo_wf.phase.flatten()[0] - start_wf.phase.flatten()[0])
        assert phase_shift < 1e-9, f'index:{0}, diff:{phase_shift}, phase_holo:{holo_wf.phase.flatten()[0]}, phase_start:{start_wf.phase.flatten()[0]}, amp_optim:{amplitude.flatten()[0]}, amp_holo:{holo_wf.amplitude.flatten()[0]}. amp_start:{start_wf.amplitude.flatten()[0]}'
    else:
        phase_shift = (holo_wf.phase - start_wf.phase).flatten()
        for i, d in enumerate(phase_shift):
            assert phase_shift[i] < 1e-9, f'index:{i}, diff:{d}, phase_holo:{holo_wf.phase.flatten()[i]}, phase_start:{start_wf.phase.flatten()[i]}, amp_optim:{amplitude.flatten()[i]}, amp_holo:{holo_wf.amplitude.flatten()[i]}. amp_start:{start_wf.amplitude.flatten()[i]}'


def write_summary(writer, holo, recon_wf, target_amp, iter, context, loss=None, lr=None, prefix='Iterations', show_holo="none", plane_idx=0, batch_idx=0):

    if context.write_all_planes:
        writer.add_images(f'{prefix}/Reconstructed intensities', np.expand_dims(recon_wf.intensity[recon_wf.roi][batch_idx], axis=1), iter, dataformats='NCHW')
    else:
        writer.add_image(f'{prefix}/Reconstructed intensity', recon_wf.intensity[recon_wf.roi][batch_idx][plane_idx], iter, dataformats='HW')

    if show_holo == "both" or show_holo == "amp":
        writer.add_image(f'{prefix}/Hologram amplitude', opt.Wavefront.to_numpy(holo.amplitude)[recon_wf.roi][batch_idx][0], iter, dataformats='HW')
    if show_holo == "both" or show_holo == "phase":
        writer.add_image(f'{prefix}/Hologram phase', opt.Wavefront.to_numpy(holo.phase)[recon_wf.roi][batch_idx][0], iter, dataformats='HW')

    loss = loss if loss is not None else context.loss_fn(recon_wf, target_amp)
    writer.add_scalar(f'{prefix}/Loss', loss, iter)
    # fixme scale
    writer.add_scalar(f"{prefix}/Acc", context.acc_fn(recon_wf, target_amp), iter)

    writer.add_scalar(f'{prefix}/ssim', ssim(recon_wf.normalised_amplitude[recon_wf.roi][batch_idx], target_amp[recon_wf.roi][0]), iter)  # scaling to avoid error
    writer.add_scalar(f'{prefix}/psnr', psnr(recon_wf.normalised_amplitude[recon_wf.roi][batch_idx], target_amp[recon_wf.roi][0]), iter)

    if lr: writer.add_scalar(f'{prefix}/lr', lr, iter)


def write_time_average_sequence(writer, recon_wf_stack: opt.Wavefront, target_amp, context):
    prefix = 'Time multiplexing'
    for t in range(0, recon_wf_stack.batch):  # fixme redundant computation
        recon_wf = recon_wf_stack.time_average(t_end=t+1)
        if context.write_all_planes:
            writer.add_images(f'{prefix}/TA Intensity sequence', np.expand_dims(recon_wf.intensity[recon_wf.roi][0], axis=1), t, dataformats='NCHW')
        else:
            writer.add_image(f'{prefix}/TA Intensity sequence', recon_wf.intensity[recon_wf.roi][0][0], t, dataformats='HW')

        writer.add_scalar(f'{prefix}/MSE', context.loss_fn(recon_wf, target_amp), t)
        writer.add_scalar(f'{prefix}/SSIM', ssim(recon_wf.normalised_amplitude[recon_wf.roi], target_amp[recon_wf.roi]), t)
        writer.add_scalar(f'{prefix}/PSNR', psnr(recon_wf.normalised_amplitude[recon_wf.roi], target_amp[recon_wf.roi]), t)
