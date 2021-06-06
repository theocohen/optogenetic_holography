import configargparse


class ArgParser():

    def __init__(self):
        self.p = configargparse.ArgumentParser()
        self.p.add_argument('-c', '--config_path', is_config_file=True, help='Config file path.')

    def _add_io_args(self):
        self.p.add_argument('--target_path', required=True, type=str, help='')
        self.p.add_argument('--padding', type=int, nargs='+', default=0, help='Scalar or [left, right, top, bottom]')
        self.p.add_argument('--optimize_resolution', type=bool, default=False, help='')
        self.p.add_argument('--output_path', type=str, default='./output/', help='')

    def _add_propagation_args(self):
        self.p.add_argument('--start_wf_intensity', type=float, default=1, help='')
        self.p.add_argument('--start_wf_phases', type=str, choices=['flat', 'random'], default='flat', help='')
        self.p.add_argument('--wavelength', type=float, default=488, help='unit: nm')
        self.p.add_argument('--pixel_pitch', type=float, default=10, help='unit: um')
        self.p.add_argument('--propagation_model', type=str, default='FourierFresnel', choices=['FourierFresnel', 'Fresnel'],
                       help='')
        self.p.add_argument('--propagation_dist', type=float, nargs='+', default=10,
                       help='Scalar or [start, end] from which linear space will be constructed. unit: cm')
        self.p.add_argument('--lens_radius', type=float, default=6, help='unit: mm')
        self.p.add_argument('--lens_focal_length', type=float, default=20, help='unit: cm')

    def _add_method_hyperparams(self):
        self.p.add_argument('--method', type=str, default='bin_amp_phase_GS', help='',
                       choices=['bin_amp_phase_GS', 'bin_amp_amp_GS', 'bin_amp_phase_SGD', 'bin_amp_amp_SGD',
                                'bin_amp_amp_sig_SGD'], )
        self.p.add_argument('--time_averaging_batch', type=int, default=1, help='')
        self.p.add_argument('--iterations', type=int, default=10, help='')
        self.p.add_argument('--lr', type=float, default=0.01, help='')
        self.p.add_argument('--bin_amp_modulation', type=str, default='otsu', help='',
                       choices=['otsu', 'yen', 'isodata', 'li', 'minimum', 'mean', 'niblack', 'sauvola', 'triangle'])
        self.p.add_argument('--random_holo_init', type=bool, default=True, help='')
        self.p.add_argument('--scale_loss', type=bool, default=False, help='')

    def _add_logging_params(self):
        self.p.add_argument('--remove_airy_disk', type=bool, default=False, help='')
        self.p.add_argument('--crop_roi', type=bool, default=True, help='')
        self.p.add_argument('--summary_freq', type=int, default=10, help='')
        self.p.add_argument('--run_comment', type=str, default='', help='')

    def parse_all_args(self):
        self._add_io_args()
        self._add_propagation_args()
        self._add_method_hyperparams()
        self._add_logging_params()
        return self.p.parse_args()