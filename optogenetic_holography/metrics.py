import torch


class MSE(torch.nn.Module):
    def __init__(self, mask=None, average_batch_grads=True, normalise_recon=False):
        super(MSE, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.average_batch_grads = average_batch_grads
        self.mask = mask
        self.normalise_recon = normalise_recon

    def forward(self, recon_wf, target_amp, force_average=False):
        if self.normalise_recon:
            loss = self.criterion(recon_wf.normalised_amplitude[recon_wf.roi], target_amp[recon_wf.roi])
        else:
            loss = self.criterion(recon_wf.amplitude[recon_wf.roi].detach(), target_amp[recon_wf.roi])
        if self.mask is not None:
            loss *= self.mask

        if self.average_batch_grads or force_average:
            loss = loss.mean(dim=(0, 1, 2, 3), keepdim=False)
        else:
            loss = loss.mean(dim=(1, 2, 3), keepdim=False)

        return loss