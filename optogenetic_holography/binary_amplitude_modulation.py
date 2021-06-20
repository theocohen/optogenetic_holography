from torch.autograd import Function

class BinAmpMod(Function):

    @staticmethod
    def forward(ctx, phase):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.phase = phase
        return (phase > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output