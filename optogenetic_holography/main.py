import time
import matplotlib
from metrics import MSE, Accuracy
from scale_optimiser import ScaleOptimiser

matplotlib.use('agg')
import os
import logging
from shutil import copy2

import numpy as np
import torch

from optogenetic_holography.arg_parser import ArgParser
from optogenetic_holography.optics import optics_backend as opt
from optogenetic_holography.utils import init_writer, write_time_average_sequence, config_logger, load_mask
import optogenetic_holography.algorithms as algorithms


def main():
    args, param_groups = ArgParser().parse_all_args()

    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Init wavefronts
    target_wf = opt.Wavefront.from_images(args.target_path, optimize_resolution=args.optimize_resolution, padding=args.padding, scale_intensity=args.target_wf_intensity, device=device)
    start_wf = opt.Wavefront(target_wf.resolution, roi=target_wf.roi, scale_intensity=args.start_wf_intensity, device=device, target_mean_amp=target_wf.amplitude.mean())
    if args.start_wf_phases == 'random':
        start_wf = opt.RandomPhaseMask().forward(start_wf)

    # Tensorboard writer
    dim = '3D' if target_wf.depth > 1 else '2D'
    run_dir = f"{args.propagation_model}/{dim}/{args.method}"
    if param_groups['plot_params'].full_plot_name:
        param_groups['plot_params'].plot_name = f"{args.propagation_model}-{dim}-{args.method}"
    writer, summary_dir = init_writer(args.output_path, run_dir, setup=args.comment)
    if args.config_path:
        copy2(args.config_path, os.path.join(summary_dir, 'config.txt'))  # copy config file to summary directory

    # propagation model
    if len(args.propagation_dist) == 2:
        assert dim == '3D', "Specified non-scalar propagation distance but only target is not 3D"
        args.propagation_dist = np.linspace(args.propagation_dist[0], args.propagation_dist[1], num=target_wf.depth)

    if args.propagation_model == 'FourierFresnel':
        propagator = opt.FourierFresnelPropagator(args.lens_radius, args.lens_focal_length, args.wavelength, args.pixel_pitch, args.propagation_dist, remove_airy_disk=args.remove_airy_disk)
    if args.propagation_model == 'Fresnel':
        propagator = opt.FresnelPropagator(args.wavelength, args.pixel_pitch, args.propagation_dist)
    if args.propagation_model == 'Fourier':
        assert dim == '2D', "Fourier model can only be used for single plane target"
        propagator = opt.FourierLensPropagator(args.lens_radius, args.lens_focal_length, remove_airy_disk=args.remove_airy_disk)

    # setting logger format
    config_logger(summary_dir, run_dir)
    logging.info(f'PyTorch device: {device}')
    if device.type == 'cuda':
        print(torch.cuda.get_device_properties(device))

    """
    def vectorised_loss(input, target):
        return torch.nn.MSELoss(reduction='none')(input, target).mean(dim=(1, 2, 3), keepdim=False)
    loss_fn = torch.nn.MSELoss() if args.average_batch_grads else vectorised_loss
    """
    mask = load_mask(args.target_mask_path, device)
    loss_fn = MSE(mask=mask, average_batch_grads=args.average_batch_grads, normalise_recon=args.normalise_recon).to(device)
    acc_fn = Accuracy(mask=mask).to(device)
    scale_fn = ScaleOptimiser(target_wf.amplitude, loss_fn, summary_dir)
    param_groups['method_params'].loss_fn = loss_fn
    param_groups['method_params'].acc_fn = acc_fn
    param_groups['method_params'].scale_fn = scale_fn

    # methods
    generator = getattr(algorithms, args.method)

    # main program
    start_time = time.time()
    logging.info("Starting")

    before_bin_holo_wf, holo_wf = generator(start_wf, target_wf.amplitude, propagator, writer, param_groups['method_params'])

    holo_wf.plot(summary_dir, param_groups['plot_params'], type='intensity', title='holo', is_holo=True)

    recon_wf_stack = propagator.forward(holo_wf)
    write_time_average_sequence(writer, recon_wf_stack, target_wf.amplitude, param_groups['method_params'])
    recon_wf = recon_wf_stack.time_average()

    scale = scale_fn(recon_wf)
    recon_wf.plot(summary_dir, param_groups['plot_params'], type='intensity', title='recon', mask=mask, scale=scale)

    recon_acc = acc_fn(recon_wf, target_wf.amplitude, scale=scale)
    loss = loss_fn(recon_wf, target_wf.amplitude, scale=scale)
    logging.info(f"\nAccuracy for (scaled) recon wf = {recon_acc}")
    logging.info(f"Loss for (scaled) recon wf = {loss}\n")

    if args.plot_before_bin:
        holo_type = 'phase' if 'phase' in args.method else 'intensity'
        before_bin_recon_wf_stack = propagator.forward(before_bin_holo_wf)  # for reference
        before_bin_holo_wf.plot(summary_dir, param_groups['plot_params'], type=holo_type, title='before_bin-holo', is_holo=True)
        before_bin_recon_wf = before_bin_recon_wf_stack.time_average()

        scale = scale_fn(before_bin_recon_wf)
        before_bin_recon_wf.plot(summary_dir, param_groups['plot_params'], type='intensity', title='before_bin-recon', mask=mask, scale=scale)

        recon_acc = acc_fn(before_bin_recon_wf, target_wf.amplitude, scale=scale)
        loss = loss_fn(before_bin_recon_wf, target_wf.amplitude, scale=scale)
        logging.info(f"Accuracy for (scaled) before bin recon wf = {recon_acc}")
        logging.info(f"MSE for (scaled) before bin recon wf = {loss}")

    logging.info(f"Finished in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - start_time))}\n")

    writer.close()


if __name__ == '__main__':
    main()