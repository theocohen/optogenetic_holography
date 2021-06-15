import torch
from torch import optim
import matplotlib.pyplot as plt
import logging


class ScaleOptimiser():

    def __init__(self, target_amp, criterion, dir, iterations=100, lr=0.1):
        self.target_amp = target_amp
        self.criterion = criterion
        self.iterations = iterations
        self.lr = lr
        self.dir = dir

    def find_scale(self, recon_wf, wf_name='', init_scale=None):
        losses = []
        scales = []

        if init_scale:
            scale = torch.tensor(float(init_scale))
        else:
            scale = recon_wf.target_mean_amp if (recon_wf.target_mean_amp is not None) else torch.tensor(1.0)
        log_scale = torch.log(scale).reshape(recon_wf.batch, 1, 1, 1).requires_grad_(True).to(recon_wf.device)

        optimizer = optim.Adam([{'params': log_scale}], lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        for iter in range(self.iterations):
            optimizer.zero_grad()
            scale = torch.exp(log_scale)
            loss = self.criterion(recon_wf, self.target_amp, force_average=True, scale=scale)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            losses.append(loss.clone().cpu().detach().numpy())
            scales.append(scale.clone().cpu().detach().numpy())

        scale = scale.detach()
        logging.info(f"Optimal scale {scale.squeeze().cpu().numpy()} for {wf_name} ")
        plt.scatter(scales, losses)
        plt.xlabel('Intensity scale')
        plt.ylabel('MSE between target and scaled recon')
        plt.savefig(f"{self.dir}/optimal_scale_{wf_name}.jpg")
        plt.close()

        return scale

