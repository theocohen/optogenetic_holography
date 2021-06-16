import torch
from torch import optim
import matplotlib.pyplot as plt
import logging


class ScaleOptimiser(torch.nn.Module):

    def __init__(self, criterion, dir, iterations=100, lr=0.1):
        super(ScaleOptimiser, self).__init__()
        self.criterion = criterion
        self.iterations = iterations
        self.lr = lr
        self.dir = dir

    def forward(self, recon_wf, target_amp, plot_title=None, init_scale=None):
        recon_wf_copy = recon_wf.copy(copy_u=True, detach=True)
        losses = []
        scales = []

        scale = torch.tensor(float(init_scale)) if init_scale else torch.tensor(1)
        log_scale = torch.log(scale).to(recon_wf.device).requires_grad_(True)

        optimizer = optim.Adam([{'params': log_scale}], lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        for iter in range(self.iterations):
            optimizer.zero_grad()
            scale = torch.exp(log_scale)
            loss = self.criterion(recon_wf_copy, target_amp, force_average=True, scale=scale)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            losses.append(loss.clone().cpu().detach().numpy())
            scales.append(scale.clone().cpu().detach().numpy())

        scale = scale.detach()
        if plot_title is not None:
            plt.scatter(scales, losses)
            plt.xlabel('Intensity scale')
            plt.ylabel('MSE between target and scaled recon')
            plt.savefig(f"{self.dir}/optimal_scale_{plot_title}.jpg")
            plt.close()

        return scale

