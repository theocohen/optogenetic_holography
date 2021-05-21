import logging

from optogenetic_holography.optics import optics_backend as opt
from optogenetic_holography.utils import write_summary


def gercherberg_saxton(start_wf: opt.Wavefront, target_amplitude: opt.Wavefront, propagator: opt.Propagator, writer, max_iter=1000) -> opt.Wavefront:
    start_amp = start_wf.amplitude
    holo_wf = start_wf.copy(copy_wf=True)

    for iter in range(max_iter):
        recon_wf = propagator.forward(holo_wf)
        recon_wf.amplitude = target_amplitude
        holo_wf = propagator.backward(recon_wf)
        holo_wf.amplitude = start_amp

        if not iter % 100:
            logging.info("GS iteration {}/{}".format(iter, max_iter))
            write_summary(writer, holo_wf.phase, recon_wf, target_amplitude, iter)

    return holo_wf


