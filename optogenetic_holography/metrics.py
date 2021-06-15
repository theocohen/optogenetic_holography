import torch


class MSE(torch.nn.Module):
    def __init__(self, mask=None, average_batch_grads=True, normalise_recon=False):
        super(MSE, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.average_batch_grads = average_batch_grads
        self.mask = mask
        self.normalise_recon = normalise_recon

    def forward(self, recon_wf, target_amp, scale=1, force_average=False):
        if scale is None:
            scale = torch.tensor(1).reshape(recon_wf.batch, 1, 1, 1)
        if self.normalise_recon:  #fixme fails
            loss = self.criterion(recon_wf.normalised_amplitude[recon_wf.roi], target_amp[recon_wf.roi])
        else:
            loss = self.criterion(scale * (recon_wf.amplitude[recon_wf.roi] ** 2), target_amp[recon_wf.roi] ** 2)
        if self.mask is not None:
            loss *= self.mask

        if self.average_batch_grads or force_average:
            loss = loss.mean(dim=(0, 1, 2, 3), keepdim=False)
        else:
            loss = loss.mean(dim=(1, 2, 3), keepdim=False)

        return loss


class Accuracy(torch.nn.Module):

    def __init__(self, mask=None):
        super(Accuracy, self).__init__()
        self.mask = mask

    def forward(self, recon_wf, target_amp, scale=1):
        if scale is None:
            scale = torch.tensor(1).reshape(recon_wf.batch, 1, 1, 1)
        target_int = torch.square(target_amp[recon_wf.roi])
        recon_int = scale * torch.square(recon_wf.amplitude[recon_wf.roi])
        if self.mask:
            recon_int *= self.mask

        return torch.sum(recon_int * target_int) / torch.sqrt(torch.sum(target_int ** 2) * torch.sum(recon_int ** 2))

