import logging
from collections import defaultdict

import torch
from torch import optim
from torch.nn.functional import mse_loss

from optogenetic_holography.binarization import from_amp_to_bin_amp, from_phase_to_bin_amp
from optogenetic_holography.optics import optics_backend as opt
from optogenetic_holography.utils import write_summary, assert_phase_unchanged


def bin_amp_phase_mgsa(start_wf, target_amplitude, propagator, writer, context):
    """Technically not the original Gercherberg-Saxton algorithm as not restricted to Fourier propagation"""

    holo_wf = start_wf.copy(copy_u=True, batch=context.ta_batch, depth=target_amplitude.shape[1])  #  optimize hologram stack
    #holo_wf = start_wf.copy(copy_u=True, batch=context.ta_batch, depth=1)  # in-loop
    if context.random_holo_init:
        holo_wf.set_random_phase()

    for iter in range(context.iterations):
        recon_wf = propagator.forward(holo_wf)

        if not iter % context.summary_freq:
            logging.info("MGSA iteration {}/{}".format(iter, context.iterations))
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, context, show_holo=context.write_holo)

        recon_wf.amplitude = target_amplitude
        #recon_wf.set_amplitude(target_amplitude, mask=context.loss_fn.mask)
        #holo_wf.phase = propagator.backward(recon_wf).phase.mean(dim=1, keepdim=True)  # in-loop mean
        holo_wf.phase = propagator.backward(recon_wf).phase  # holo stack
        #holo_wf.amplitude = from_phase_to_bin_amp(propagator.backward(recon_wf).phase)  binarisation in-loop

    #binarization
    before_bin_holo_wf = holo_wf.copy(copy_u=True)

    holo_wf.depth = 1
    holo_wf.polar_to_rect(from_phase_to_bin_amp(holo_wf.phase.mean(dim=1, keepdim=True)), start_wf.phase)
    #holo_wf.polar_to_rect(from_phase_to_bin_amp(holo_wf.phase), start_wf.phase)

    recon_wf = propagator.forward(holo_wf)
    write_summary(writer, holo_wf, recon_wf, target_amplitude, iter + 1, context, show_holo=context.write_holo)

    return before_bin_holo_wf, holo_wf


def bin_amp_amp_mgsa(start_wf, target_amplitude, propagator, writer, context):
    holo_wf = start_wf.copy(copy_u=True, batch=context.ta_batch, depth=target_amplitude.shape[1])
    if context.random_holo_init:
        holo_wf.set_random_amplitude()

    for iter in range(context.iterations):
        recon_wf = propagator.forward(holo_wf)

        if not iter % context.summary_freq:
            logging.info("MGSA iteration {}/{}".format(iter, context.iterations))
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, context, show_holo='none')

        recon_wf.amplitude = target_amplitude
        #recon_wf.set_amplitude(target_amplitude, mask=context.loss_fn.mask)

        holo_wf.amplitude = propagator.backward(recon_wf).amplitude
        #holo_wf.amplitude = from_amp_to_bin_amp(propagator.backward(recon_wf), method=context.bin_amp_mod)  binarisation in-loop

    before_bin_holo_wf = holo_wf.copy(copy_u=True)

    holo_wf.depth = 1
    holo_wf.polar_to_rect(from_amp_to_bin_amp(holo_wf.amplitude.mean(dim=1, keepdim=True), method=context.bin_amp_mod), start_wf.phase)

    recon_wf = propagator.forward(holo_wf)
    write_summary(writer, holo_wf, recon_wf, target_amplitude, iter + 1, context, show_holo=context.write_holo)

    return before_bin_holo_wf, holo_wf


def bin_amp_phase_sgd(start_wf, target_amplitude, propagator, writer, context):
    holo_wf = start_wf.copy(copy_u=True, batch=context.ta_batch)
    if context.random_holo_init:
        holo_wf.set_random_phase()

    phase = holo_wf.phase.requires_grad_(True)
    params = [{'params': phase}]
    optimizer = optim.Adam(params, lr=context.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for iter in range(context.iterations):

        optimizer.zero_grad()

        holo_wf.polar_to_rect(start_wf.amplitude, phase)
        recon_wf = propagator.forward(holo_wf)

        loss = context.loss_fn(recon_wf, target_amplitude)
        if context.average_batch_grads:
            loss.backward()
        else:
            loss.backward(torch.ones_like(loss))
            loss = loss.mean()
        scheduler.step(loss)
        optimizer.step()

        if not iter % context.summary_freq:
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"SGD iteration {iter}/{context.iterations}. Loss {loss}, lr {lr}")
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, context, loss=loss, lr=context.lr, show_holo=context.write_holo)

    with torch.no_grad():
        before_bin_holo_wf = holo_wf.copy(copy_u=True)
        holo_wf.polar_to_rect(from_phase_to_bin_amp(holo_wf.phase), start_wf.phase)

        recon_wf = propagator.forward(holo_wf)
        loss = context.loss_fn(recon_wf, target_amplitude, force_average=True)
        write_summary(writer, holo_wf, recon_wf, target_amplitude, iter + 1, context, loss=loss, lr=lr, show_holo=context.write_holo)
    return before_bin_holo_wf, holo_wf


def bin_amp_amp_sgd(start_wf, target_amplitude, propagator, writer, context):

    holo_wf = start_wf.copy(copy_u=True, batch=context.ta_batch)
    if context.random_holo_init:
        holo_wf.set_random_amplitude()

    amplitude = holo_wf.amplitude
    log_amplitude = torch.log(amplitude).requires_grad_(True)

    params = [{'params': log_amplitude}]
    optimizer = optim.Adam(params, lr=context.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for iter in range(context.iterations):

        optimizer.zero_grad()
        amplitude = torch.exp(log_amplitude)
        holo_wf.polar_to_rect(amplitude, start_wf.phase)

        #assert_phase_unchanged(amplitude, holo_wf, start_wf, just_check_first=True)  # comment for perf

        recon_wf = propagator.forward(holo_wf)

        loss = context.loss_fn(recon_wf, target_amplitude)
        if context.average_batch_grads:
            loss.backward()
        else:
            loss.backward(torch.ones_like(loss))
            loss = loss.mean()
        scheduler.step(loss)
        optimizer.step()

        if not iter % context.summary_freq:
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"SGD iteration {iter}/{context.iterations}. Loss {loss}, lr {lr}")
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, context, loss=loss, lr=lr, show_holo="none")

    with torch.no_grad():
        before_bin_holo_wf = holo_wf.copy(copy_u=True)

        holo_wf.amplitude = from_amp_to_bin_amp(holo_wf.amplitude, method=context.bin_amp_mod)

        recon_wf = propagator.forward(holo_wf)
        loss = context.loss_fn(recon_wf, target_amplitude, force_average=True)
        write_summary(writer, holo_wf, recon_wf, target_amplitude, iter + 1, context, loss=loss, lr=lr, show_holo=context.write_holo)
    return before_bin_holo_wf, holo_wf


def bin_amp_amp_sig_sgd(start_wf, target_amplitude, propagator, writer, context):
    # fixme
    holo_wf = start_wf.copy(copy_u=True, batch=context.ta_batch)
    if context.random_holo_init:
        holo_wf.set_random_amplitude()

    amplitude = holo_wf.amplitude.requires_grad_(True)

    params = [{'params': amplitude}]
    optimizer = optim.Adam(params, lr=context.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for iter in range(context.iterations):

        optimizer.zero_grad()

        # approximation to heavyside step function
        # todo learn optimal sharpness and threshold
        threshold = context.bin_threshold if context.bin_threshold is not None else amplitude.mean()
        bin_amp = 1 / (1 + torch.exp(- context.bin_sharpness * (amplitude - threshold)))

        holo_wf.polar_to_rect(bin_amp, start_wf.phase)

        #assert_phase_unchanged(bin_amp, holo_wf, start_wf, just_check_first=True)  # comment for perf

        recon_wf = propagator.forward(holo_wf)

        loss = context.loss_fn(recon_wf, target_amplitude)
        if context.average_batch_grads:
            loss.backward()
        else:
            loss.backward(torch.ones_like(loss))
            loss = loss.mean()
        scheduler.step(loss)
        optimizer.step()

        if not iter % context.summary_freq:
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"SGD iteration {iter}/{context.iterations}. Loss {loss}, lr {lr}")
            write_summary(writer, holo_wf, recon_wf, target_amplitude, iter, context, loss=loss, lr=lr, show_holo=context.write_holo)

    with torch.no_grad():
        before_bin_holo_wf = holo_wf.copy(copy_u=True)
        holo_wf.amplitude = from_amp_to_bin_amp(holo_wf.amplitude, method="otsu")

        recon_wf = propagator.forward(holo_wf)
        loss = context.loss_fn(recon_wf, target_amplitude, force_average=True)
        write_summary(writer, holo_wf, recon_wf, target_amplitude, iter + 1, context, loss=loss, lr=lr)
    return before_bin_holo_wf, holo_wf


