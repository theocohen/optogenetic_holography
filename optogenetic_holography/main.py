import time
import matplotlib
matplotlib.use('agg')
import os
import logging
import sys

import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from optogenetic_holography.arg_parser import ArgParser
from optogenetic_holography.optics import optics_backend as opt
from optogenetic_holography.utils import mkdir, init_writer
import optogenetic_holography.algorithms as algorithms

logging.basicConfig(format='%(asctime)s - %(run_dir)s - %(message)s', level=logging.INFO)
#logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def main():
    args, method_params = ArgParser().parse_all_args()

    # Init wavefronts
    target_wf = opt.Wavefront.from_images(args.target_path, optimize_resolution=args.optimize_resolution, padding=args.padding, scale_intensity=args.target_wf_intensity)
    start_wf = opt.Wavefront(target_wf.resolution, roi=target_wf.roi, scale_intensity=args.start_wf_intensity)
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

    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.run_dir = run_dir
        return record
    logging.setLogRecordFactory(record_factory)

    summary_dir = os.path.join(args.output_path, 'summaries', run_dir)
    writer = init_writer(args.output_path, run_dir, setup=args.comment)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #todo
    def vectorised_loss(input, target):
        return nn.MSELoss(reduction='none')(input, target).mean(dim=(1, 2, 3), keepdim=False)
    loss_fn = nn.MSELoss() if args.average_batch_grads else vectorised_loss
    method_params.loss_fn = loss_fn

    # methods
    generator = getattr(algorithms, args.method)

    # main program
    start_time = time.time()
    logging.info("Starting")

    holo_wf = generator(start_wf, target_wf.amplitude, propagator, writer, method_params)
    #holo_wf.plot(intensity=defaultdict(str, title=experiment + "__holo", path=output_path, save=True))

    recon_wf_stack = propagator.forward(holo_wf)
    #plot_time_average_sequence(writer, recon_wf_stack, target_wf.amplitude)
    recon_wf = recon_wf_stack.time_average()
    #recon_wf.plot(intensity=defaultdict(str, title=experiment + "__recon", path=output_path, save=True, suppress_center=suppress_center, normalize=True, threshold_foreground=True,crop_roi=True))

    logging.info(f"Finished in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - start_time))}\n")
    writer.close()


if __name__ == '__main__':
    main()