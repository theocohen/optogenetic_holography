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


def write_summary(writer, holo, recon_wf, target_amp, iter, loss=None, lr=None, prefix='test'):
    """todo ROI"""

    recon_amp = recon_wf.amplitude.detach()
    # scaling from neural-holo (legit?)
    #recon_amp *= torch.sum(recon_amp * target_amp) / recon_wf.total_intensity
    recon_amp /= recon_amp.max()

    writer.add_image(f'{prefix}/Reconstructed intensity', opt.Wavefront.to_numpy(recon_amp ** 2), iter, dataformats='HW')
    writer.add_image(f'{prefix}/Hologram pattern', opt.Wavefront.to_numpy(holo), iter, dataformats='HW')

    writer.add_scalar(f'{prefix}/Loss', loss if loss is not None else mse_loss(recon_amp, target_amp), iter)
    writer.add_scalar(f'{prefix}/ssim', ssim(recon_amp, target_amp), iter)
    writer.add_scalar(f'{prefix}/psnr', psnr(recon_amp, target_amp), iter)

    if lr:
        writer.add_scalar(f'{prefix}/lr', lr, iter)

