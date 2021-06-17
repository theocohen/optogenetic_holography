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
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

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
    if not os.path.exists(path):
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


def write_summary(writer, holo, recon_wf, target_amp, iter, context, loss=None, lr=None, prefix='Iterations', show_holo="none", plane_idx=0, batch_idx=0, scale=None):

    if scale is None:
        scale = context.scale_fn(recon_wf, target_amp) if context.write_with_scale else torch.tensor(1)

    if context.write_images:
        if context.write_all_planes:
            writer.add_images(f'{prefix}/Reconstructed intensities', opt.Wavefront.to_numpy(scale) * np.expand_dims(recon_wf.intensity[recon_wf.roi][batch_idx], axis=1), iter, dataformats='NCHW')
        else:
            writer.add_image(f'{prefix}/Reconstructed intensity', opt.Wavefront.to_numpy(scale) * recon_wf.intensity[recon_wf.roi][batch_idx][plane_idx], iter, dataformats='HW')

        if show_holo == "both" or show_holo == "amp":
            writer.add_image(f'{prefix}/Hologram amplitude', opt.Wavefront.to_numpy(holo.amplitude)[recon_wf.roi][batch_idx][0], iter, dataformats='HW')
        if show_holo == "both" or show_holo == "phase":
            writer.add_image(f'{prefix}/Hologram phase', opt.Wavefront.to_numpy(holo.phase)[recon_wf.roi][batch_idx][0], iter, dataformats='HW')

    loss = loss if loss is not None else context.loss_fn(recon_wf, target_amp, scale=scale)

    writer.add_scalar(f'{prefix}/Scale', scale, iter)
    writer.add_scalar(f'{prefix}/Loss', loss, iter)
    writer.add_scalar(f"{prefix}/Acc", context.acc_fn(recon_wf, target_amp, scale), iter)
    writer.add_scalar(f'{prefix}/ssim', ssim(recon_wf.normalised_amplitude[recon_wf.roi][batch_idx], target_amp[recon_wf.roi][0]), iter)  # scaling to avoid error
    writer.add_scalar(f'{prefix}/psnr', psnr(recon_wf.normalised_amplitude[recon_wf.roi][batch_idx], target_amp[recon_wf.roi][0]), iter)

    if lr: writer.add_scalar(f'{prefix}/lr', lr, iter)


def write_time_average_sequence(writer, recon_wf_stack: opt.Wavefront, target_amp, context, scale=1):
    # time averaged metrics
    prefix = 'Time multiplexing'
    for t in range(0, recon_wf_stack.batch):  # fixme redundant computation
        recon_wf = recon_wf_stack.time_average(t_end=t+1)
        if context.write_images:
            if context.write_all_planes:
                writer.add_images(f'{prefix}/TA Intensity sequence', scale * np.expand_dims(recon_wf.intensity[recon_wf.roi][0], axis=1), t, dataformats='NCHW')
            else:
                writer.add_image(f'{prefix}/TA Intensity sequence', recon_wf.intensity[recon_wf.roi][0][0], t, dataformats='HW')

        writer.add_scalar(f'{prefix}/MSE', context.loss_fn(recon_wf, target_amp, scale=scale), t)
        writer.add_scalar(f'{prefix}/SSIM', ssim(recon_wf.normalised_amplitude[recon_wf.roi], target_amp[recon_wf.roi]), t)
        writer.add_scalar(f'{prefix}/PSNR', psnr(recon_wf.normalised_amplitude[recon_wf.roi], target_amp[recon_wf.roi]), t)


def write_batch_summary_to_csv(summary_dir, recon_wf_stack, target_amp, context, method_name, scale=1, modulation="final"):
    # fixme assumes same scale for all batch
    # sequential metrics
    loss_bach = opt.Wavefront.to_numpy(context.loss_fn(recon_wf_stack, target_amp, scale=scale, force_batch_reduct='none'))
    acc_batch = opt.Wavefront.to_numpy(context.acc_fn(recon_wf_stack, target_amp, scale=scale, force_batch_reduct='none'))
    write_metrics_to_csv(summary_dir, method_name, modulation, acc_batch, loss_bach)


def write_metrics_to_csv(dir, method_name, modulation, accuracy, loss):
    file_path = dir + "/metrics.csv"
    df = pd.DataFrame({"method": method_name, "mod": modulation, "acc": accuracy, "loss": loss})
    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)


def assert_phase_unchanged(holo_wf, start_wf, threshold=1e-7):
    # prevent phase modulation
    assert (holo_wf.phase[holo_wf.amplitude != 0] - start_wf.phase.broadcast_to(holo_wf.shape)[holo_wf.amplitude != 0]).max() < threshold