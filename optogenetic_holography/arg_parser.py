import configargparse
import numpy as np

from optogenetic_holography.optics import optics_backend as opt


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


class ArgParser():


    def __init__(self):
        self.p = configargparse.ArgumentParser()
        self.p.add_argument('-c', '--config_path', is_config_file=True, help='Config file path.')

    def _add_io_args(self):
        self.p.add_argument('--target_path', required=True, type=str, help='')
        self.p.add_argument('--target_mask_path', type=str, default='', help='')
        self.p.add_argument('--padding', type=int, nargs='+', default=[0], help='Scalar or [left, right, top, bottom]')
        self.p.add_argument('--optimize_resolution', type=str2bool, nargs='?', default=False, help='')
        self.p.add_argument('--output_path', type=str, default='./output/', help='')
        self.p.add_argument('--method', type=str, default='bin_amp_phase_GS', help='',
                            choices=['bin_amp_phase_mgsa', 'bin_amp_amp_mgsa', 'bin_amp_phase_sgd', 'bin_amp_amp_sgd',
                                     'bin_amp_amp_sig_sgd'], )
        self.p.add_argument('--comment', type=str, default='', help='')

    def _add_propagation_args(self):
        self.p.add_argument('--target_wf_intensity', type=float, default=1, help='')
        self.p.add_argument('--start_wf_intensity', type=float, default=1, help='')
        self.p.add_argument('--start_wf_phases', type=str, choices=['flat', 'random'], default='flat', help='')
        self.p.add_argument('--wavelength', type=float, default=488, help='unit: nm')
        self.p.add_argument('--pixel_pitch', type=float, default=10, help='unit: um')
        self.p.add_argument('--propagation_model', type=str, default='FourierFresnel', choices=['FourierFresnel', 'Fresnel', 'Fourier'],
                       help='')
        self.p.add_argument('--propagation_dist', type=float, nargs='+', default=[10],
                       help='Scalar or [start, end] from which linear space will be constructed. unit: cm')
        self.p.add_argument('--lens_radius', type=float, default=6, help='unit: mm')
        self.p.add_argument('--lens_focal_length', type=float, default=20, help='unit: cm')
        self.p.add_argument('--remove_airy_disk', type=str2bool, nargs='?', default=False, help='')

    def _add_method_params(self):
        g = self.p.add_argument_group('method_params', '')
        g.add_argument('--ta_batch', type=int, default=1, help='Time averaging batch size')
        g.add_argument('--iterations', type=int, default=10, help='')
        g.add_argument('--lr', type=float, default=0.1, help='For sgd methods only')
        g.add_argument('--bin_amp_mod', type=str, default='otsu', help='for bin_amp_amp methods only',
                       choices=['otsu', 'yen', 'isodata', 'li', 'minimum', 'mean', 'niblack', 'sauvola', 'triangle'])
        g.add_argument('--bin_sharpness', type=float, help='For bin_amp_amp_sig_sgd only')
        g.add_argument('--bin_threshold', type=float, help='For bin_amp_amp_sig_sgd only')
        g.add_argument('--random_holo_init', type=str2bool, nargs='?', default=True, help='')
        g.add_argument('--average_batch_grads', type=str2bool, nargs='?', default=True, help='')
        g.add_argument('--learn_scale', type=str, default='implicit',
                       help='For SGD only. Either optimise scale as part of main optimisation or with an explicit optimiser',
                       choices=['implicit', 'explicit', 'none'])

        g.add_argument('--summary_freq', type=int, default=10, help='')
        g.add_argument('--write_all_planes', type=str2bool, nargs='?', default=False, help='')
        g.add_argument('--write_holo', type=str, default='none', help='')
        g.add_argument('--write_with_scale', type=str2bool, nargs='?', default=True, help='Scale images and metrics to best match target when writing summary')

    def _add_logging_params(self):
        g = self.p.add_argument_group('plot_params', '')
        g.add_argument('--crop_roi',type=str2bool, nargs='?', default=True, help='')
        g.add_argument('--cmap', type=str, default='gray', help='')
        g.add_argument('--normalise_plot', type=str2bool, nargs='?', default=False, help='')
        g.add_argument('--threshold_foreground', type=str2bool, nargs='?', default=True, help='')
        g.add_argument('--masked_plot', type=str2bool, nargs='?', default=False, help='')
        g.add_argument('--figsize', type=int, nargs=2, default=None, help='')
        g.add_argument('--full_plot_name', type=str2bool, nargs='?', default=False, help='')
        g.add_argument('--plot_before_bin', type=str2bool, nargs='?', default=False, help='')
        g.add_argument('--scale_plot_to_target', type=str2bool, nargs='?', default=False, help='')

    def _adjust_units(self, args):
        args.wavelength = args.wavelength * opt.nm
        args.pixel_pitch = (args.pixel_pitch * opt.um,) * 2
        args.lens_radius = args.lens_radius * opt.mm
        args.lens_focal_length = args.lens_radius * opt.cm

        if len(args.propagation_dist) > 2:
            self.p.error('--prop_dist either accepts one distance or a tuple of the form [start, stop].')
            self.p.error('--prop_dist either accepts one distance or a tuple of the form [start, stop].')
        args.propagation_dist = np.array(args.propagation_dist) * opt.cm

        return args

    def parse_all_args(self):
        self._add_io_args()
        self._add_propagation_args()
        self._add_method_params()
        self._add_logging_params()
        args = self.p.parse_args()
        args = self._adjust_units(args)

        param_groups = {}
        for group in self.p._action_groups:
            if group.title in ['method_params', 'plot_params']:
                param_groups[group.title] = configargparse.Namespace(**{a.dest: getattr(args, a.dest, None) for a in group._group_actions})

        return args, param_groups