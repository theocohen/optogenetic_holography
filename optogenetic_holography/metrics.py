import torch


class MSE(torch.nn.Module):
    def __init__(self, mask=None, average_batch_grads=True):
        super(MSE, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.average_batch_grads = average_batch_grads
        self.mask = mask

    def forward(self, recon_wf, target_amp, scale=None, force_batch_reduct=''):
        recon_int = (recon_wf.amplitude[recon_wf.roi] ** 2)
        if scale is not None:
            recon_int *= scale
        loss = self.criterion(recon_int, target_amp[recon_wf.roi] ** 2)
        if self.mask is not None:
            loss *= self.mask

        if force_batch_reduct == 'avg' or (self.average_batch_grads and force_batch_reduct != 'none'):
            loss = loss.mean(dim=(0, 1, 2, 3), keepdim=False)
        else:
            loss = loss.mean(dim=(1, 2, 3), keepdim=False)

        return loss


class Accuracy(torch.nn.Module):

    def __init__(self, mask=None):
        super(Accuracy, self).__init__()
        self.mask = mask

    def forward(self, recon_wf, target_amp, scale=1, force_batch_reduct=''):
        if scale is None:
            scale = torch.tensor(1).reshape(recon_wf.batch, 1, 1, 1)
        target_int = torch.square(target_amp[recon_wf.roi])
        recon_int = scale * torch.square(recon_wf.amplitude[recon_wf.roi])
        if self.mask is not None:
            recon_int *= self.mask

        dim = (1, 2, 3) if force_batch_reduct == 'none' else (0, 1, 2, 3)
        return torch.sum(recon_int * target_int, dim) / torch.sqrt(torch.sum(target_int ** 2, dim) * torch.sum(recon_int ** 2, dim))

