import os
#from pytorch_ssim import ssim
from datetime import datetime

import torch
from piq import psnr, ssim
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter

from optogenetic_holography.optics import optics_backend as opt


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def init_writer(output_path, experiment):
    summaries_dir = os.path.join(output_path + 'summaries/' + experiment + '/' + datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    cond_mkdir(summaries_dir)
    return SummaryWriter(summaries_dir)


def write_summary(writer, holo, recon_wf, target_amp, iter, loss=None, lr=None, prefix='', scale_loss=False, modulation="amp"):
    """todo ROI"""

    # scaling from neural-holo (legit?)
    #scaled_recon_amp = recon_amp * torch.sum(recon_amp * target_amp) / recon_wf.total_intensity
    #scaled_recon_amp = recon_amp / recon_amp.max()  # scaling

    writer.add_image(f'{prefix}/Reconstructed intensity', recon_wf.intensity[0], iter, dataformats='HW')

    if modulation == "both" or holo == "amp":
        writer.add_image(f'{prefix}/Hologram amplitude', opt.Wavefront.to_numpy(holo.amplitude)[0], iter, dataformats='HW')
    if modulation == "both" or holo == "phase":
        writer.add_image(f'{prefix}/Hologram phase', opt.Wavefront.to_numpy(holo.phase)[0], iter, dataformats='HW')

    loss = loss if loss is not None else (mse_loss(recon_wf.scaled_amplitude.detach(), target_amp) if scale_loss else mse_loss(recon_wf.amplitude.detach(), target_amp))
    writer.add_scalar(f'{prefix}/Loss', loss, iter)

    writer.add_scalar(f'{prefix}/ssim', ssim(recon_wf.scaled_amplitude, target_amp), iter)  # scaling to avoid error
    writer.add_scalar(f'{prefix}/psnr', psnr(recon_wf.scaled_amplitude, target_amp), iter)

    if lr:
        writer.add_scalar(f'{prefix}/lr', lr, iter)

