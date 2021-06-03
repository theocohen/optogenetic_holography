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


def init_writer(output_path, experiment, setup=None):
    summaries_dir = output_path + 'summaries/' + experiment + '/'
    summaries_dir += setup if setup is not None else datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    cond_mkdir(summaries_dir)
    return SummaryWriter(summaries_dir)


def assert_phase_unchanged(amplitude, holo_wf, start_wf, just_check_first=True):
    if just_check_first:
        phase_shift = (holo_wf.phase.flatten()[0] - start_wf.phase.flatten()[0])
        assert phase_shift < 1e-9, f'index:{0}, diff:{phase_shift}, phase_holo:{holo_wf.phase.flatten()[0]}, phase_start:{start_wf.phase.flatten()[0]}, amp_optim:{amplitude.flatten()[0]}, amp_holo:{holo_wf.amplitude.flatten()[0]}. amp_start:{start_wf.amplitude.flatten()[0]}'
    else:
        phase_shift = (holo_wf.phase - start_wf.phase).flatten()
        for i, d in enumerate(phase_shift):
            assert phase_shift[i] < 1e-9, f'index:{i}, diff:{d}, phase_holo:{holo_wf.phase.flatten()[i]}, phase_start:{start_wf.phase.flatten()[i]}, amp_optim:{amplitude.flatten()[i]}, amp_holo:{holo_wf.amplitude.flatten()[i]}. amp_start:{start_wf.amplitude.flatten()[i]}'


def write_summary(writer, holo, recon_wf, target_amp, iter, loss=None, lr=None, prefix='', scale_loss=False, show_holo="none"):
    """todo ROI"""

    # scaling from neural-holo (legit?)
    #scaled_recon_amp = recon_amp * torch.sum(recon_amp * target_amp) / recon_wf.total_intensity
    #scaled_recon_amp = recon_amp / recon_amp.max()  # scaling

    writer.add_image(f'{prefix}/Reconstructed intensity', recon_wf.intensity[0], iter, dataformats='HW')

    if show_holo == "both" or show_holo == "amp":
        writer.add_image(f'{prefix}/Hologram amplitude', opt.Wavefront.to_numpy(holo.amplitude)[0], iter, dataformats='HW')
    if show_holo == "both" or show_holo == "phase":
        writer.add_image(f'{prefix}/Hologram phase', opt.Wavefront.to_numpy(holo.phase)[0], iter, dataformats='HW')

    loss = loss if loss is not None else (mse_loss(recon_wf.scaled_amplitude.detach(), target_amp) if scale_loss else mse_loss(recon_wf.amplitude.detach(), target_amp))
    writer.add_scalar(f'{prefix}/Loss', loss, iter)

    writer.add_scalar(f'{prefix}/ssim', ssim(recon_wf.scaled_amplitude, target_amp), iter)  # scaling to avoid error
    writer.add_scalar(f'{prefix}/psnr', psnr(recon_wf.scaled_amplitude, target_amp), iter)

    if lr:
        writer.add_scalar(f'{prefix}/lr', lr, iter)

