import logging
from collections import defaultdict
import time

import torch
from torch import optim
from torch.nn.functional import mse_loss

from optogenetic_holography.binarization import from_amp_to_bin_amp, from_phase_to_bin_amp
from optogenetic_holography.optics import optics_backend as opt
from optogenetic_holography.utils import write_summary, assert_phase_unchanged


def bin_amp_phase_gercherberg_saxton(start_wf: opt.Wavefront, target_amplitude, propagator: opt.Propagator, writer, max_iter=1000, scale_loss=False, summary_freq=100, batch=1) -> opt.Wavefront:
    """Technically not the original Gercherberg-Saxton algorithm as not restricted to Fourier propagation"""
    start_time = time.time()

    #holo_wf = start_wf.copy(copy_u=True, batch=batch, depth=target_amplitude.shape[1])
    holo_wf = start_wf.copy(copy_u=True, batch=batch, depth=1)  # in-loop
    holo_wf.set_random_phase()

    for iter in range(max_iter):
        recon_wf = propagator.forward(holo_wf)

        if not iter % summary_freq:
            logging.info("GS iteration {}/{}".format(iter, max_iter))
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, prefix='', scale_loss=scale_loss, show_holo="none")

        recon_wf.amplitude = target_amplitude
        holo_wf.phase = propagator.backward(recon_wf).phase.mean(dim=1, keepdim=True)  #fixme optimal to average in-loop?
        #holo_wf.amplitude = from_phase_to_bin_amp(propagator.backward(recon_wf))  binarisation in-loop

    #binarization
    holo_wf.polar_to_rect(from_phase_to_bin_amp(holo_wf), start_wf.phase)
    recon_wf = propagator.forward(holo_wf)
    write_summary(writer, holo_wf, recon_wf, target_amplitude, iter + 1, prefix='', scale_loss=scale_loss, show_holo="none")

    logging.info(f"Finished in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - start_time))}")
    return holo_wf


def bin_amp_amp_gercherberg_saxton(start_wf: opt.Wavefront, target_amplitude, propagator: opt.Propagator, writer, max_iter=1000, scale_loss=False, bin_amp_mode="otsu", summary_freq=100, batch=1) -> opt.Wavefront:
    start_time = time.time()

    holo_wf = start_wf.copy(copy_u=True, batch=batch, depth=target_amplitude.shape[1])
    holo_wf.set_random_amplitude()

    for iter in range(max_iter):
        recon_wf = propagator.forward(holo_wf)

        if not iter % summary_freq:
            logging.info("GS iteration {}/{}".format(iter, max_iter))
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, prefix='', scale_loss=scale_loss, show_holo="none")

        recon_wf.amplitude = target_amplitude
        holo_wf.amplitude = propagator.backward(recon_wf).amplitude
        #holo_wf.amplitude = from_amp_to_bin_amp(propagator.backward(recon_wf), method=bin_amp_mode)  binarisation in-loop

    holo_wf.depth = 1
    holo_wf.polar_to_rect(from_amp_to_bin_amp(holo_wf.amplitude.mean(dim=1, keepdim=True), method=bin_amp_mode), start_wf.phase)

    recon_wf = propagator.forward(holo_wf)
    write_summary(writer, holo_wf, recon_wf, target_amplitude, iter + 1, prefix='', scale_loss=scale_loss, show_holo="amp")
    logging.info(f"Finished in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - start_time))}")

    return holo_wf


def phase_sgd(start_wf, target_amplitude, propagator, loss_fn, writer, max_iter=1000, lr=0.1, scale_loss=False, summary_freq=100, batch=1) -> opt.Wavefront:
    holo_wf = start_wf.copy(copy_u=True)
    holo_wf.set_random_phase()

    phase = start_wf.phase.requires_grad_(True)
    params = [{'params': phase}]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for iter in range(max_iter):

        optimizer.zero_grad()

        holo_wf.polar_to_rect(start_wf.amplitude, phase)
        recon_wf = propagator.forward(holo_wf)

        recon_amp = recon_wf.amplitude / recon_wf.amplitude.max() if scale_loss else recon_wf.amplitude
        loss = loss_fn(recon_amp[recon_wf.roi], target_amplitude[recon_wf.roi])
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if not iter % summary_freq:
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"SGD iteration {iter}/{max_iter}. Loss {loss}, lr {lr}")
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, loss=loss, lr=lr, prefix='', show_holo="phase")

    return holo_wf


def bin_amp_phase_sgd(start_wf, target_amplitude, propagator, loss_fn, writer, max_iter=1000, lr=0.1, scale_loss=False, summary_freq=100, batch=1) -> opt.Wavefront:
    start_time = time.time()
    holo_wf = start_wf.copy(copy_u=True)
    holo_wf.set_random_phase()

    phase = holo_wf.phase.requires_grad_(True)
    params = [{'params': phase}]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for iter in range(max_iter):

        optimizer.zero_grad()

        holo_wf.polar_to_rect(start_wf.amplitude, phase)
        recon_wf = propagator.forward(holo_wf)

        recon_amp = recon_wf.amplitude / recon_wf.amplitude.max() if scale_loss else recon_wf.amplitude
        loss = loss_fn(recon_amp[recon_wf.roi], target_amplitude[recon_wf.roi])
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if not iter % summary_freq:
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"SGD iteration {iter}/{max_iter}. Loss {loss}, lr {lr}")
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, loss=loss, lr=lr, prefix='', show_holo="none")

    with torch.no_grad():
        holo_wf.polar_to_rect(from_phase_to_bin_amp(holo_wf), start_wf.phase)

        recon_wf = propagator.forward(holo_wf)
        recon_amp = recon_wf.amplitude / recon_wf.amplitude.max() if scale_loss else recon_wf.amplitude
        loss = loss_fn(recon_amp[recon_wf.roi], target_amplitude[recon_wf.roi])
        write_summary(writer, holo_wf, recon_wf, target_amplitude, iter + 1, loss=loss, lr=lr, prefix='', show_holo="none")
        logging.info(f"Finished in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - start_time))}")
    return holo_wf


def bin_amp_amp_sgd(start_wf, target_amplitude, propagator, loss_fn, writer, max_iter=1000, lr=0.1, scale_loss=False, bin_amp_mode="otsu", summary_freq=100, batch=1) -> opt.Wavefront:
    start_time = time.time()

    holo_wf = start_wf.copy(copy_u=True, batch=batch)
    #holo_wf.set_random_amplitude()

    amplitude = holo_wf.amplitude
    log_amplitude = torch.log(amplitude).requires_grad_(True)

    params = [{'params': log_amplitude}]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for iter in range(max_iter):

        optimizer.zero_grad()
        amplitude = torch.exp(log_amplitude)
        holo_wf.polar_to_rect(amplitude, start_wf.phase)

        assert_phase_unchanged(amplitude, holo_wf, start_wf, just_check_first=True)  # comment for perf

        recon_wf = propagator.forward(holo_wf)

        recon_amp = recon_wf.amplitude / recon_wf.amplitude.max() if scale_loss else recon_wf.amplitude
        loss = loss_fn(recon_amp[recon_wf.roi], target_amplitude[recon_wf.roi])
        #loss.backward(torch.ones_like(loss))
        #optimizer.step()
        #scheduler.step(loss.mean())
        loss = mse_loss(recon_amp[recon_wf.roi], target_amplitude[recon_wf.roi])
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if not iter % summary_freq:
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"SGD iteration {iter}/{max_iter}. Loss {loss}, lr {lr}")
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, loss=loss, lr=lr, prefix='', show_holo="none")

    with torch.no_grad():
        holo_wf.amplitude = from_amp_to_bin_amp(holo_wf.amplitude, method=bin_amp_mode)

        recon_wf = propagator.forward(holo_wf)
        recon_amp = recon_wf.amplitude / recon_wf.amplitude.max() if scale_loss else recon_wf.amplitude
        loss = mse_loss(recon_amp[recon_wf.roi], target_amplitude[recon_wf.roi])
        write_summary(writer, holo_wf, recon_wf, target_amplitude, iter + 1, loss=loss, lr=lr, prefix='', show_holo='none')
        logging.info(f"Finished in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - start_time))}")
    return holo_wf


def bin_amp_amp_sig_sgd(start_wf, target_amplitude, propagator, loss_fn, writer, max_iter=1000, lr=0.1, scale_loss=False, sharpness=1, threshold=None, summary_freq=100, batch=1) -> opt.Wavefront:
    # fixme
    start_time = time.time()
    holo_wf = start_wf.copy(copy_u=True)
    holo_wf.set_random_amplitude()

    amplitude = holo_wf.amplitude.requires_grad_(True)

    params = [{'params': amplitude}]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for iter in range(max_iter):

        optimizer.zero_grad()

        # approximation to heavyside step function
        # todo learn optimal sharpness and threshold
        threshold = threshold if threshold is not None else amplitude.mean()
        bin_amp = 1 / (1 + torch.exp(- sharpness * (amplitude - threshold)))

        holo_wf.polar_to_rect(bin_amp, start_wf.phase)

        assert_phase_unchanged(bin_amp, holo_wf, start_wf, just_check_first=True)  # comment for perf

        recon_wf = propagator.forward(holo_wf)

        recon_amp = recon_wf.amplitude / recon_wf.amplitude.max() if scale_loss else recon_wf.amplitude
        loss = loss_fn(recon_amp[recon_wf.roi], target_amplitude[recon_wf.roi])
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if not iter % summary_freq:
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"SGD iteration {iter}/{max_iter}. Loss {loss}, lr {lr}")
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, loss=loss, lr=lr, prefix='', show_holo="both")

    with torch.no_grad():
        holo_wf.amplitude = from_amp_to_bin_amp(holo_wf.amplitude, method="otsu")

        recon_wf = propagator.forward(holo_wf)
        recon_amp = recon_wf.amplitude / recon_wf.amplitude.max() if scale_loss else recon_wf.amplitude
        loss = loss_fn(recon_amp[recon_wf.roi], target_amplitude[recon_wf.roi])
        write_summary(writer, holo_wf, recon_wf, target_amplitude, iter + 1, loss=loss, lr=lr, prefix='')
        logging.info(f"Finished in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - start_time))}")
    return holo_wf


