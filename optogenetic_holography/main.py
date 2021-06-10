import time
import matplotlib
matplotlib.use('agg')
import os
import logging
from shutil import copy2

import numpy as np
import torch

from optogenetic_holography.arg_parser import ArgParser
from optogenetic_holography.optics import optics_backend as opt
from optogenetic_holography.utils import mkdir, init_writer, plot_time_average_sequence, config_logger
import optogenetic_holography.algorithms as algorithms


def main():
    args, param_groups = ArgParser().parse_all_args()

    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Init wavefronts
    target_wf = opt.Wavefront.from_images(args.target_path, optimize_resolution=args.optimize_resolution, padding=args.padding, scale_intensity=args.target_wf_intensity, device=device)
    start_wf = opt.Wavefront(target_wf.resolution, roi=target_wf.roi, scale_intensity=args.start_wf_intensity, device=device)
    if args.start_wf_phases == 'random':
        start_wf = opt.RandomPhaseMask().forward(start_wf)

    # propagation model
    if isinstance(args.propagation_dist, list):
        args.propagation_dist = np.linspace(args.propagation_dist[0], args.propagation_dist[1], num=target_wf.depth)
    args.propagation_dist *= opt.cm

    if args.propagation_model == 'FourierFresnel':
        propagator = opt.FourierFresnelPropagator(args.lens_radius, args.lens_focal_length, args.wavelength, args.pixel_pitch, args.propagation_dist)
    if args.propagation_model == 'Fresnel':
        propagator = opt.FresnelPropagator(args.wavelength, args.pixel_pitch, args.propagation_dist)

    # Tensorboard writer
    dim = '3D' if target_wf.depth > 1 else '2D'
    run_dir = f"{args.propagation_model}/{dim}/{args.method}"
    writer, summary_dir = init_writer(args.output_path, run_dir, setup=args.comment)
    if args.config_path:
        copy2(args.config_path, os.path.join(summary_dir, 'config.txt'))  # copy config file to summary directory

    # setting logger format
    config_logger(summary_dir, run_dir)
    logging.info(f'PyTorch device: {device}')

    def vectorised_loss(input, target):
        return torch.nn.MSELoss(reduction='none')(input, target).mean(dim=(1, 2, 3), keepdim=False)
    loss_fn = torch.nn.MSELoss() if args.average_batch_grads else vectorised_loss
    param_groups['method_params'].loss_fn = loss_fn.to(device)

    # methods
    generator = getattr(algorithms, args.method)

    # main program
    start_time = time.time()
    logging.info("Starting")

    holo_wf = generator(start_wf, target_wf.amplitude, propagator, writer, param_groups['method_params'])
    holo_wf.plot(summary_dir, param_groups['plot_params'], type='intensity', title='holo')

    recon_wf_stack = propagator.forward(holo_wf)
    plot_time_average_sequence(writer, recon_wf_stack, target_wf.amplitude)

    recon_wf = recon_wf_stack.time_average()
    recon_wf.plot(summary_dir, param_groups['plot_params'], type='intensity', title='recon')

    logging.info(f"Finished in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - start_time))}\n")

    writer.close()


if __name__ == '__main__':
    main()