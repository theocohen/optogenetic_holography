import logging
from collections import defaultdict

import torch
from torch import optim

from optogenetic_holography.optics import optics_backend as opt
from optogenetic_holography.utils import write_summary, assert_phase_unchanged


def bin_amp_phase_gercherberg_saxton(start_wf: opt.Wavefront, target_amplitude, propagator: opt.Propagator, bin_amp_modulation, writer, max_iter=1000, scale_loss=False) -> opt.Wavefront:
    """Technically not the original Gercherberg-Saxton algorithm as not restricted to Fourier propagation"""
    holo_wf = start_wf.copy(copy_wf=True)

    for iter in range(max_iter):
        recon_wf = propagator.forward(holo_wf)

        if not iter % 100:
            logging.info("GS iteration {}/{}".format(iter, max_iter))
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, prefix='', scale_loss=scale_loss, modulation="both")

        recon_wf.amplitude = target_amplitude
        holo_wf.phase = propagator.backward(recon_wf).phase.mean(dim=0)

    #binarization
    holo_wf.polar_to_rect(bin_amp_modulation(holo_wf), start_wf.phase)  # fixme
    recon_wf = propagator.forward(holo_wf)
    write_summary(writer, holo_wf, recon_wf, target_amplitude, iter + 1, prefix='', scale_loss=scale_loss, modulation="both")

    return holo_wf


def bin_amp_amp_gercherberg_saxton(start_wf: opt.Wavefront, target_amplitude, propagator: opt.Propagator, bin_amp_modulation, writer, max_iter=1000, scale_loss=False) -> opt.Wavefront:
    holo_wf = start_wf.copy(copy_wf=True)

    for iter in range(max_iter):
        recon_wf = propagator.forward(holo_wf)

        if not iter % 100:
            logging.info("GS iteration {}/{}".format(iter, max_iter))
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, prefix='', scale_loss=scale_loss, modulation="amp")

        recon_wf.amplitude = target_amplitude
        holo_wf.amplitude = propagator.backward(recon_wf).amplitude

    holo_wf.amplitude = bin_amp_modulation(holo_wf, method="amplitude")

    recon_wf = propagator.forward(holo_wf)
    write_summary(writer, holo_wf, recon_wf, target_amplitude, iter + 1, prefix='', scale_loss=scale_loss, modulation="both")

    return holo_wf


def phase_sgd(start_wf, target_amplitude, propagator, loss_fn, writer, max_iter=1000, lr=0.1, scale_loss=False) -> opt.Wavefront:
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
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, loss=loss, lr=lr, prefix='', modulation="phase")

    return holo_wf


def bin_amp_phase_sgd(start_wf, target_amplitude, propagator, loss_fn, bin_amp_modulation, writer, max_iter=1000, lr=0.1, scale_loss=False) -> opt.Wavefront:
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
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, loss=loss, lr=lr, prefix='', modulation="phase")

    with torch.no_grad():
        holo_wf.polar_to_rect(bin_amp_modulation(holo_wf), start_wf.phase)  # fixme

        recon_wf = propagator.forward(holo_wf)
        recon_amp = recon_wf.amplitude / recon_wf.amplitude.max() if scale_loss else recon_wf.amplitude
        loss = loss_fn(recon_amp.double(), target_amplitude.double())
        write_summary(writer, holo_wf, recon_wf, target_amplitude, iter + 1, loss=loss, lr=lr, prefix='', modulation="phase")
    return holo_wf


def bin_amp_amp_sgd(start_wf, target_amplitude, propagator, loss_fn, bin_amp_modulation, writer, max_iter=1000, lr=0.1, scale_loss=False) -> opt.Wavefront:
    holo_wf = start_wf.copy(copy_wf=True)

    amplitude = start_wf.amplitude
    log_amplitude = torch.log(amplitude).requires_grad_(True)

    params = [{'params': log_amplitude}]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for iter in range(max_iter):

        optimizer.zero_grad()
        amplitude = torch.exp(log_amplitude)
        holo_wf.polar_to_rect(amplitude, start_wf.phase)

        assert_phase_unchanged(amplitude, holo_wf, start_wf, just_check_first=True)

        recon_wf = propagator.forward(holo_wf)

        recon_amp = recon_wf.amplitude / recon_wf.amplitude.max() if scale_loss else recon_wf.amplitude
        loss = loss_fn(recon_amp.double(), target_amplitude.double())
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if not iter % 100:
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"SGD iteration {iter}/{max_iter}. Loss {loss}, lr {lr}")
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, loss=loss, lr=lr, prefix='', modulation="both")

    with torch.no_grad():
        holo_wf.amplitude = bin_amp_modulation(holo_wf, method="amplitude")

        recon_wf = propagator.forward(holo_wf)
        recon_amp = recon_wf.amplitude / recon_wf.amplitude.max() if scale_loss else recon_wf.amplitude
        loss = loss_fn(recon_amp.double(), target_amplitude.double())
        write_summary(writer, holo_wf, recon_wf, target_amplitude, iter + 1, loss=loss, lr=lr, prefix='')
    return holo_wf


"""
def bin_amp_sgd(start_wf, target_amplitude, propagator, loss_fn, bin_amp_modulation, writer, max_iter=1000, lr=0.1, scale_loss=False) -> opt.Wavefront:
    # Binary Gradient
    holo_wf = start_wf.copy(copy_wf=True)

    phase = start_wf.phase.requires_grad_(True)
    params = [{'params': phase}]
    optimizer = optim.Adam(params, lr=1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for iter in range(max_iter):

        optimizer.zero_grad()

        #bin_amp_mod = phase.heaviside(torch.tensor([0.0]))

        bin_phase = phase.sign().detach().requires_grad_()
        bin_amp_mod = (bin_phase + 1) / 2

        holo_wf.polar_to_rect(bin_amp_mod, start_wf.phase)
        recon_wf = propagator.forward(holo_wf)

        recon_amp = recon_wf.amplitude / recon_wf.amplitude.max() if scale_loss else recon_wf.amplitude
        loss = loss_fn(recon_amp.double(), target_amplitude.double())
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if not iter % 100:
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"SGD iteration {iter}/{max_iter}. Loss {loss}, lr {lr}")
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, loss=loss, lr=lr, prefix='')
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, loss=loss, lr=lr, prefix='bin_amp')

    return holo_wf
"""


