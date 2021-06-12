import torch

class MSE(torch.nn.Module):
    def __init__(self, mask=None, average_batch_grads=True, ):
        super(MSE, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.average_batch_grads = average_batch_grads
        self.mask = mask

    def forward(self, input, target):

        loss = self.criterion(input, target)
        if self.mask is not None:
            loss *= self.mask

        if self.average_batch_grads:
            loss = loss.mean(dim=(0, 1, 2, 3), keepdim=False)
        else:
            loss = loss.mean(dim=(1, 2, 3), keepdim=False)

        return loss