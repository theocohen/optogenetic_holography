import copy
import glob
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
plt.style.use('dark_background')

#from optogenetic_holography.optics.wavefront import WavefrontInterface


class Wavefront:
    def __init__(self, resolution, depth=1, batch=1, roi=None, scale_intensity=1, device='cpu'):
        assert len(resolution) == 2
        self.resolution = resolution
        self.shape = (batch, depth,) + resolution
        self.u = torch.ones(self.shape, dtype=torch.complex128).to(device) * scale_intensity ** (1/2)
        self.roi = roi if roi is not None else (slice(None),) * 4
        self.device = device

    @classmethod
    def from_images(cls, path, intensity=True, scale_intensity=1, padding=0, optimize_resolution=True, device='cpu'):
        """assumes all image are of same shape"""
        images = np.stack([cv2.imread(file, 0) for file in sorted(glob.glob(path))])

        resolution = images.shape[1:]
        if isinstance(padding, int):
            padding = [padding] * 4  # [left, right, top, bottom]
        resolution = (resolution[0] + padding[0] + padding[1], resolution[1] + padding[2] + padding[3])
        if optimize_resolution:
            resolution = 2 ** np.ceil(np.log2(resolution))  # powers of 2 for optimized FFT
            resolution = (int(resolution[0]), int(resolution[1]))

        wf = Wavefront(resolution, depth=images.shape[0], device=device)
        wf.roi = slice(None), slice(None), slice(padding[0], padding[0] + images.shape[1]), slice(padding[2], padding[2] + images.shape[2])
        padded_images = np.zeros(images.shape[:1] + resolution, dtype=np.uint8)
        padded_images[wf.roi[1:]] = images

        if intensity:
            wf.intensity = torch.tensor(padded_images * scale_intensity).double()
            wf.amplitude /= wf.amplitude.amax(dim=(2, 3), keepdim=True)
        else:
            wf.phase = torch.tensor(padded_images)
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
    def scaled_amplitude(self):
        return self.amplitude / self.amplitude.amax(dim=(2,3), keepdim=True)

    @amplitude.setter
    def amplitude(self, new_amplitude):
        self.u = torch.polar(new_amplitude.broadcast_to(self.shape).to(self.device), self.phase)

    def set_random_amplitude(self):
        self.amplitude = torch.rand(self.shape).double()

    @property
    def phase(self):
        return self.u.angle()

    def set_random_phase(self):
        self.phase = (np.pi * (1 - 2 * torch.rand(self.shape))).double()  # between -pi to pi

    @phase.setter
    def phase(self, new_phase):
        self.u = torch.polar(self.amplitude, new_phase.broadcast_to(self.shape).to(self.device))

    @property
    def intensity(self):
        return Wavefront.to_numpy(torch.square(self.amplitude))

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

    def plot(self, dir, options, type='intensity', title=''):
        if type == 'intensity':
            img = self.intensity
            if options.remove_airy_disk:
                img[:, self.resolution[0] // 2, self.resolution[1] // 2] = 0
            if options.normalise_int:
                img = self.intensity / self.intensity.max()
        elif type == 'phase':
            img = Wavefront.to_numpy(self.phase)

        if options.crop_roi and self.roi is not None:
            img = img[self.roi]
        if options.threshold_foreground:
            img = (img > filters.threshold_otsu(img))

        for t in range(self.batch):
            for d in range(self.depth):
                plt.imshow(img[t][d], cmap=options.cmap)
                plt.xticks([]), plt.yticks([])
                plt.savefig(f"{dir}/{title}_t{str(t+1)}_d{str(d+1)}.jpg", bbox_inches="tight", pad_inches = 0)
                plt.close()

    def plot_old(self, **kwargs):
        if 'intensity' in kwargs:
            options = kwargs['intensity']
            img = self.intensity
            if options['suppress_center']:
                img[:, self.resolution[0] // 2, self.resolution[1] // 2] = 0
            if options['normalize']:
                img = self.intensity / self.intensity.max()
        elif 'phase' in kwargs:
            img = Wavefront.to_numpy(self.phase)
            options = kwargs['phase']

        if options["crop_roi"] and self.roi is not None:
            img = img[self.roi]
        if options["threshold_foreground"]:
            img = (img > filters.threshold_otsu(img))
        for t in range(self.batch):
            for d in range(self.depth):
                plt.imshow(img[t][d], cmap='gray')
                plt.xticks([]), plt.yticks([])
                if options['save']: plt.savefig(f"{options['path'] + options['title']}_t{str(t+1)}_d{str(d+1)}.jpg", bbox_inches="tight", pad_inches = 0)
                plt.colorbar()
                plt.title(f"{options['title']}_t{str(t+1)}_d{str(d+1)}.jpg")
                plt.show()
                plt.close()

    def time_average(self, t_start=0, t_end=None):
        ta_wf = self.copy()
        ta_wf.batch = 1
        t_end = self.batch if t_end is None else t_end
        ta_wf.u = self.u[t_start:t_end, :, :].mean(dim=0, keepdim=True)
        return ta_wf

    def copy(self, copy_u=False, batch=None, depth=None):
        depth = self.depth if depth is None else depth
        batch = self.batch if batch is None else batch
        copy_wf = Wavefront(self.resolution, depth=depth, batch=batch, roi=self.roi, device=self.device)
        if copy_u:
            copy_wf.u = self.u.broadcast_to(copy_wf.shape)
        return copy_wf

    def requires_grad(self):
        self.u.requires_grad_(True)