import logging
import glob
import os

import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from skimage import filters
#plt.style.use('dark_background')

#from optogenetic_holography.optics.wavefront import WavefrontInterface


class Wavefront:
    def __init__(self, resolution, depth=1, batch=1, roi=None, scale_intensity=1, device='cpu', target_mean_amp=None):
        assert len(resolution) == 2
        self.resolution = resolution
        self.shape = (batch, depth,) + resolution
        self.u = torch.ones(self.shape, dtype=torch.complex64).to(device) * scale_intensity ** (1/2)
        self.roi = roi if roi is not None else (slice(None),) * 4
        self.device = device
        self.target_mean_amp = target_mean_amp

    @classmethod
    def from_images(cls, path, intensity=True, scale_intensity=1, padding=[0], optimize_resolution=True, device='cpu'):
        """assumes all image are of same shape"""
        images = np.stack([cv2.imread(file, 0) for file in sorted(glob.glob(path))])

        resolution = images.shape[1:]
        if len(padding) == 1:
            padding = padding * 4  # [left, right, top, bottom]
        resolution = (resolution[0] + padding[0] + padding[1], resolution[1] + padding[2] + padding[3])
        if optimize_resolution:
            resolution = 2 ** np.ceil(np.log2(resolution))  # powers of 2 for optimized FFT
            resolution = (int(resolution[0]), int(resolution[1]))

        wf = Wavefront(resolution, depth=images.shape[0], device=device)
        wf.roi = slice(None), slice(None), slice(padding[0], padding[0] + images.shape[1]), slice(padding[2], padding[2] + images.shape[2])
        padded_images = np.zeros(images.shape[:1] + resolution, dtype=np.uint8)
        padded_images[wf.roi[1:]] = images

        if intensity:
            wf.intensity = torch.tensor(padded_images * scale_intensity / 255).float()
        else:
            wf.phase = torch.tensor(padded_images)  # fixme map to -pi to pi
        return wf

    @property
    def batch(self):
        return self.shape[0]

    @batch.setter
    def batch(self, new_batch):
        prev_batch = self.batch
        self.shape = (new_batch,) + self.shape[1:]
        if new_batch > prev_batch:
            self.u = self.u.broadcast_to(self.shape)

    @property
    def depth(self):
        return self.shape[1]

    @depth.setter
    def depth(self, new_depth):
        prev_depth = self.depth
        self.shape = (self.shape[0], new_depth) + self.shape[2:]
        if new_depth > prev_depth:
            self.u = self.u.broadcast_to(self.shape)

    @property
    def amplitude(self):
        return self.u.abs()

    @property
    def normalised_amplitude(self):
        return self.amplitude / self.amplitude.max()

    def scaled_amplitude(self):
        return self.amplitude * self.target_mean_amp / self.amplitude.mean(dim=(1, 2, 3))

    @amplitude.setter
    def amplitude(self, new_amplitude):
        self.u = torch.polar(new_amplitude.broadcast_to(self.shape).to(self.device), self.phase)

    def set_amplitude(self, new_amplitude, mask=None):  #fixme
        if mask is None:
            self.amplitude = new_amplitude
        else:
            self.amplitude[self.roi] = ~mask.bool() * self.amplitude[self.roi] + mask * new_amplitude[self.roi]

    def set_random_amplitude(self):
        self.amplitude = torch.rand(self.shape).float()

    @property
    def phase(self):
        return self.u.angle()

    def set_random_phase(self):
        self.phase = (np.pi * (1 - 2 * torch.rand(self.shape))).float()  # between -pi to pi

    @phase.setter
    def phase(self, new_phase):
        self.u = torch.polar(self.amplitude, new_phase.broadcast_to(self.shape).to(self.device))

    @property
    def intensity(self):
        return Wavefront.to_numpy(torch.square(self.amplitude))

    def scaled_intensity(self):
        scale = self.target_mean_amp / self.amplitude.mean(dim=(1, 2, 3))
        return scale, Wavefront.to_numpy(torch.square(self.amplitude * scale))

    @intensity.setter
    def intensity(self, new_intensity):
        self.amplitude = torch.sqrt(new_intensity).broadcast_to(self.shape).to(self.device)

    @property
    def total_intensity(self):
        return Wavefront.to_numpy(torch.sum(torch.square(self.amplitude), (-2,-1)))

    @classmethod
    def to_numpy(cls, tt):
        return tt.cpu().detach().numpy()

    def polar_to_rect(self, amp, phase):
        """from neural holo"""
        self.u = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase)).broadcast_to(self.shape).to(self.device)

    def assert_equal(self, other_field, atol=1e-6):
        return torch.allclose(self.u, other_field.u, atol=atol)

    def plot(self, dir, options, type='intensity', title='', mask=None, scale=1, is_holo=False, force_colorbar=False):
        if not os.path.exists(dir):
            os.makedirs(dir)
        if type == 'intensity':
            img = scale * self.amplitude ** 2
            if options.normalise_plot and not is_holo:
                # normalising by total intensity
                img /= img.sum()
            img = Wavefront.to_numpy(img)
        elif type == 'phase':
            img = Wavefront.to_numpy(self.phase)
        if options.crop_roi and self.roi is not None:
            img = img[self.roi]
            if options.masked_plot and mask is not None:
                img *= mask.cpu().numpy()
        if options.threshold_foreground:
            img = (img > filters.threshold_otsu(img))

        plot_name = options.plot_name + '-' if options.plot_name is not None else ''

        dpi = 80
        figsize = tuple(options.figsize) if options.figsize is not None else (self.resolution[1] / float(dpi), self.resolution[0] / float(dpi))
        for t in range(self.batch):
            for d in range(self.depth):

                fig = plt.figure(figsize=figsize)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis('off')
                im = ax.imshow(img[t][d], cmap=('gray' if is_holo else options.cmap))
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                if force_colorbar or (options.plot_colorbar and not is_holo):
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax)
                plt.savefig(f"{dir}/{plot_name}{title}-t{str(t+1)}-d{str(d+1)}.jpg",bbox_inches='tight', pad_inches=0, dpi=dpi)
                plt.close()

    def time_average(self, t_start=0, t_end=None):
        # time average intensity (not amplitude!) (and discard phase)
        # fixme numerical precision
        ta_wf = self.copy(batch=1)
        t_end = self.batch if t_end is None else t_end
        ta_wf.u = torch.sqrt((self.amplitude[t_start:t_end, :, :] ** 2).mean(dim=0, keepdim=True)).type('torch.ComplexFloatTensor').to(self.device)
        return ta_wf

    def copy(self, copy_u=False, batch=None, depth=None, detach=False):
        depth = self.depth if depth is None else depth
        batch = self.batch if batch is None else batch
        copy_wf = Wavefront(self.resolution, depth=depth, batch=batch, roi=self.roi, device=self.device, target_mean_amp=self.target_mean_amp)
        if copy_u:
            copy_wf.u = self.u.broadcast_to(copy_wf.shape)
            if detach:
                copy_wf.detach_()
        return copy_wf

    def requires_grad_(self):
        self.u.requires_grad_(True)

    def detach_(self):
        self.u = self.u.detach()