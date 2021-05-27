import logging

from torch import optim

from optogenetic_holography.optics import optics_backend as opt
from optogenetic_holography.utils import write_summary


def phase_gercherberg_saxton_2D(start_wf: opt.Wavefront, target_amplitude: opt.Wavefront, propagator: opt.Propagator, writer, max_iter=1000, scale_loss=False) -> opt.Wavefront:
    start_amp = start_wf.amplitude
    holo_wf = start_wf.copy(copy_wf=True)

    for iter in range(max_iter):
        recon_wf = propagator.forward(holo_wf)

        if not iter % 100:
            logging.info("GS iteration {}/{}".format(iter, max_iter))
            write_summary(writer, holo_wf.phase, recon_wf, target_amplitude, iter, prefix='phase', scale_loss=scale_loss)

        recon_wf.amplitude = target_amplitude
        holo_wf = propagator.backward(recon_wf)
        holo_wf.amplitude = start_amp

    return holo_wf


def bin_amp_gercherberg_saxton_2D(start_wf: opt.Wavefront, target_amplitude: opt.Wavefront, propagator: opt.Propagator, writer, max_iter=1000, scale_loss=False) -> opt.Wavefront:
    holo_wf = start_wf.copy(copy_wf=True)

    for iter in range(max_iter):
        recon_wf = propagator.forward(holo_wf)

        if not iter % 100:
            logging.info("GS iteration {}/{}".format(iter, max_iter))
            write_summary(writer, holo_wf.amplitude, recon_wf, target_amplitude, iter, prefix='bin_amp', scale_loss=scale_loss)

        recon_wf.amplitude = target_amplitude
        holo_wf.amplitude = (propagator.backward(recon_wf).phase > 0).float()  # binary amplitude modulation

    return holo_wf


def phase_sgd_2D(start_wf, target_amplitude, propagator, loss_fn, writer, max_iter=1000, lr=0.1, scale_loss=False) -> opt.Wavefront:
    holo_wf = start_wf.copy(copy_wf=True)

    phase = start_wf.phase.requires_grad_(True)
    params = [{'params': phase}]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for iter in range(max_iter):

        optimizer.zero_grad()

        holo_wf.polar_to_rect(start_wf.amplitude, phase)
        recon_wf = propagator.forward(holo_wf)

        recon_amp = recon_wf.amplitude / recon_wf.amplitude.max() if scale_loss else recon_wf.amplitude
        loss = loss_fn(recon_amp.double(), target_amplitude.double())
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if not iter % 100:
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"SGD iteration {iter}/{max_iter}. Loss {loss}, lr {lr}")
            write_summary(writer, holo_wf.phase, recon_wf, target_amplitude, iter, loss=loss, lr=lr, prefix='phase')

    return holo_wf


def bin_amp_sgd_2D(start_wf, target_amplitude, propagator, loss_fn, writer, max_iter=1000, lr=0.1, scale_loss=False) -> opt.Wavefront:
    holo_wf = start_wf.copy(copy_wf=True)

    amplitude = start_wf.amplitude.requires_grad_(True)
    params = [{'params': amplitude}]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for iter in range(max_iter):

        optimizer.zero_grad()

        holo_wf.polar_to_rect(amplitude, start_wf.phase)
        recon_wf = propagator.forward(holo_wf)

        recon_amp = recon_wf.amplitude / recon_wf.amplitude.max() if scale_loss else recon_wf.amplitude
        loss = loss_fn(recon_amp.double(), target_amplitude.double())
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if not iter % 100:
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"SGD iteration {iter}/{max_iter}. Loss {loss}, lr {lr}")
            write_summary(writer, holo_wf.amplitude, recon_wf, target_amplitude, iter, loss=loss, lr=lr, prefix='bin_amp')

    return holo_wf



