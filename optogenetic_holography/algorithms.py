import logging
from collections import defaultdict

import torch
from torch import optim
from torch.nn.functional import mse_loss

from optogenetic_holography.binarization import from_amp_to_bin_amp, from_phase_to_bin_amp
from optogenetic_holography.optics import optics_backend as opt
from optogenetic_holography.utils import write_summary, assert_phase_unchanged


def bin_amp_phase_mgsa(start_wf, target_amp, propagator, writer, context):
    holo_wf = start_wf.copy(copy_u=True, batch=context.ta_batch, depth=target_amp.shape[1])  #  optimize hologram stack
    #holo_wf = start_wf.copy(copy_u=True, batch=context.ta_batch, depth=1)  # in-loop
    if context.random_holo_init:
        holo_wf.set_random_phase()

    for iter in range(context.iterations):
        recon_wf = propagator.forward(holo_wf)

        if not iter % context.summary_freq:
            scale = context.scale_fn(recon_wf, target_amp) if context.write_with_scale else torch.tensor(1)
            logging.info(f"MGSA iteration {iter}/{context.iterations}, scale {scale:.5f}")
            write_summary(writer, holo_wf, recon_wf, target_amp, iter, context, show_holo=context.write_holo, scale=scale)

        recon_wf.amplitude = target_amp
        #recon_wf.set_amplitude(target_amp, mask=context.loss_fn.mask)
        #holo_wf.phase = propagator.backward(recon_wf).phase.mean(dim=1, keepdim=True)  # in-loop mean
        holo_wf.phase = propagator.backward(recon_wf).phase  # holo stack
        #holo_wf.amplitude = from_phase_to_bin_amp(propagator.backward(recon_wf).phase)  binarisation in-loop

    #binarization
    before_bin_metadata = {'holo': holo_wf.copy(copy_u=True), 'last_scale': scale}

    holo_wf.depth = 1
    holo_wf.polar_to_rect(from_phase_to_bin_amp(holo_wf.phase.mean(dim=1, keepdim=True)), start_wf.phase)  # FIXME 3D in loop
    #holo_wf.polar_to_rect(from_phase_to_bin_amp(holo_wf.phase), start_wf.phase)

    return holo_wf, before_bin_metadata


def bin_amp_amp_mgsa(start_wf, target_amp, propagator, writer, context):
    holo_wf = start_wf.copy(copy_u=True, batch=context.ta_batch, depth=target_amp.shape[1])
    if context.random_holo_init:
        holo_wf.set_random_amplitude()

    for iter in range(context.iterations):
        recon_wf = propagator.forward(holo_wf)

        if not iter % context.summary_freq:
            scale = context.scale_fn(recon_wf, target_amp) if context.write_with_scale else torch.tensor(1)
            logging.info(f"MGSA iteration {iter}/{context.iterations}, scale {scale:.5f}")
            write_summary(writer, holo_wf, recon_wf, target_amp, iter, context, show_holo=context.write_holo, scale=scale)

        recon_wf.amplitude = target_amp
        #recon_wf.set_amplitude(target_amp, mask=context.loss_fn.mask)

        holo_wf.amplitude = propagator.backward(recon_wf).amplitude
        #holo_wf.amplitude = from_amp_to_bin_amp(propagator.backward(recon_wf), method=context.bin_amp_mod)  binarisation in-loop

    before_bin_metadata = {'holo': holo_wf.copy(copy_u=True), 'last_scale': scale}

    holo_wf.depth = 1
    holo_wf.polar_to_rect(from_amp_to_bin_amp(holo_wf.amplitude.mean(dim=1, keepdim=True), method=context.bin_amp_mod), start_wf.phase) # FIXME 3D in loop

    return holo_wf, before_bin_metadata


def bin_amp_phase_sgd(start_wf, target_amp, propagator, writer, context):
    holo_wf = start_wf.copy(copy_u=True, batch=context.ta_batch)
    if context.random_holo_init:
        holo_wf.set_random_phase()

    phase = holo_wf.phase.requires_grad_(True)
    params = [{'params': phase}]

    if context.learn_scale == 'none':
        scale = torch.tensor(1).to(start_wf.device)
    if context.learn_scale == 'implicit':
        log_scale = torch.tensor(0.0).to(start_wf.device)
        log_scale.requires_grad_(True)
        params.append({'params': log_scale})

    optimizer = optim.Adam(params, lr=context.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for iter in range(context.iterations):

        optimizer.zero_grad()

        holo_wf.polar_to_rect(start_wf.amplitude, phase)
        recon_wf = propagator.forward(holo_wf)

        if context.learn_scale == 'implicit':
            scale = torch.exp(log_scale)
        elif context.learn_scale == 'explicit':
            scale = context.scale_fn(recon_wf, target_amp)

        loss = context.loss_fn(recon_wf, target_amp, scale=scale)
        if context.average_batch_grads:
            loss.backward()
        else:
            loss.backward(torch.ones_like(loss))
            loss = loss.mean()

        if not iter % context.summary_freq:
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"SGD iteration {iter}/{context.iterations}. Loss {loss:.4f}, lr {lr:.4f}, scale {scale:.5f}")
            write_summary(writer, holo_wf, recon_wf, target_amp, iter, context, loss=loss, lr=context.lr, show_holo=context.write_holo, scale=scale)

        scheduler.step(loss)
        optimizer.step()

    holo_wf.detach_()
    before_bin_metadata = {'holo': holo_wf.copy(copy_u=True), 'recon_wf_stack': recon_wf, 'last_scale': scale}
    holo_wf.polar_to_rect(from_phase_to_bin_amp(holo_wf.phase), start_wf.phase)

    return holo_wf, before_bin_metadata


def bin_amp_amp_sgd(start_wf, target_amp, propagator, writer, context):
    holo_wf = start_wf.copy(copy_u=True, batch=context.ta_batch)
    if context.random_holo_init:
        holo_wf.set_random_amplitude()

    amplitude = holo_wf.amplitude
    log_amplitude = torch.log(amplitude).requires_grad_(True)

    params = [{'params': log_amplitude}]

    if context.learn_scale == 'none':
        scale = torch.tensor(1).to(start_wf.device)
    if context.learn_scale == 'implicit':
        log_scale = torch.tensor(0.0).to(start_wf.device)
        log_scale.requires_grad_(True)
        params.append({'params': log_scale})

    optimizer = optim.Adam(params, lr=context.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for iter in range(context.iterations):

        optimizer.zero_grad()
        amplitude = torch.exp(log_amplitude)
        holo_wf.polar_to_rect(amplitude, start_wf.phase)

        #assert_phase_unchanged(amplitude, holo_wf, start_wf, just_check_first=True)  # comment for perf

        recon_wf = propagator.forward(holo_wf)

        if context.learn_scale == 'implicit':
            scale = torch.exp(log_scale)
        elif context.learn_scale == 'explicit':
            scale = context.scale_fn(recon_wf, target_amp)

        loss = context.loss_fn(recon_wf, target_amp, scale=scale)
        if context.average_batch_grads:
            loss.backward()
        else:
            loss.backward(torch.ones_like(loss))
            loss = loss.mean()

        if not iter % context.summary_freq:
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"SGD iteration {iter}/{context.iterations}. Loss {loss:.4f}, lr {lr:.4f}, scale {scale:.5f}")
            write_summary(writer, holo_wf, recon_wf, target_amp, iter, context, loss=loss, lr=lr, show_holo=context.write_holo, scale=scale)

        scheduler.step(loss)
        optimizer.step()

    holo_wf.detach_()
    before_bin_metadata = {'holo': holo_wf.copy(copy_u=True), 'recon_wf_stack': recon_wf, 'last_scale': scale}
    holo_wf.amplitude = from_amp_to_bin_amp(holo_wf.amplitude, method=context.bin_amp_mod)

    return holo_wf, before_bin_metadata


def bin_amp_amp_sig_sgd(start_wf, target_amp, propagator, writer, context):
    # fixme
    holo_wf = start_wf.copy(copy_u=True, batch=context.ta_batch)
    if context.random_holo_init:
        holo_wf.set_random_amplitude()

    amplitude = holo_wf.amplitude.requires_grad_(True)

    params = [{'params': amplitude}]

    if context.learn_scale == 'none':
        scale = torch.tensor(1).to(start_wf.device)
    if context.learn_scale == 'implicit':
        log_scale = torch.tensor(0.0).to(start_wf.device)
        log_scale.requires_grad_(True)
        params.append({'params': log_scale})

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

        if context.learn_scale == 'implicit':
            scale = torch.exp(log_scale)
        elif context.learn_scale == 'explicit':
            scale = context.scale_fn(recon_wf, target_amp)

        loss = context.loss_fn(recon_wf, target_amp, scale=scale)
        if context.average_batch_grads:
            loss.backward()
        else:
            loss.backward(torch.ones_like(loss))
            loss = loss.mean()

        if not iter % context.summary_freq:
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"SGD iteration {iter}/{context.iterations}. Loss {loss:.4f}, lr {lr:.4f}, scale {scale:.5f}")
            write_summary(writer, holo_wf, recon_wf, target_amp, iter, context, loss=loss, lr=lr, show_holo=context.write_holo, scale=scale)

        scheduler.step(loss)
        optimizer.step()

    holo_wf.detach_()
    before_bin_metadata = {'holo': holo_wf.copy(copy_u=True), 'recon_wf_stack': recon_wf, 'last_scale': scale}
    holo_wf.amplitude = from_amp_to_bin_amp(holo_wf.amplitude, method="otsu")

    return holo_wf, before_bin_metadata


