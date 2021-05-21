import os
#from pytorch_ssim import ssim
from piq import psnr, ssim
from torch.nn.functional import mse_loss

from optogenetic_holography.optics import optics_backend as opt


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_summary(writer, holo, recon_wf, target_amp, iter):
    """todo ROI"""

    recon_amp = recon_wf.amplitude
    # scaling from neural-holo
    # recon_amp *= torch.sum(recon_amp * target_amp)
    #                            / recon_wf.total_intensity

    writer.add_image(f'Reconstructed intensity', opt.Wavefront.to_numpy(recon_amp ** 2), iter, dataformats='HW')
    writer.add_image(f'Hologram pattern', opt.Wavefront.to_numpy(holo), iter, dataformats='HW')

    writer.add_scalar(f'mse', mse_loss(recon_amp, target_amp), iter)
    writer.add_scalar(f'ssim', ssim(recon_amp, target_amp), iter)
    writer.add_scalar(f'psnr', psnr(recon_amp, target_amp), iter)