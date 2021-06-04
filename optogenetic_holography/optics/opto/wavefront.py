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
    def __init__(self, wavelength, pixel_pitch, resolution, depth=1, batch=1):
        assert len(resolution) == 2
        self.resolution = resolution
        self.depth = depth
        self.batch = batch
        self.u = torch.ones(self.shape, dtype=torch.complex128)
        self.wavelength = wavelength
        self.pixel_pitch = pixel_pitch

    @classmethod
    def from_images(cls, path, wavelength, pixel_pitch, intensity=True, scale_intensity=1, padding=10, optimize_resolution=True):
        """assumes all image are of same shape"""
        images = np.stack([cv2.imread(file, 0) for file in sorted(glob.glob(path))])

        resolution = images.shape[1:]
        if isinstance(padding, int):
            padding = [padding] * 4  # [left, right, top, bottom]
        resolution = (resolution[0] + padding[0] + padding[1], resolution[1] + padding[2] + padding[3])
        if optimize_resolution:
            resolution = 2 ** np.ceil(np.log2(resolution))  # powers of 2 for optimized FFT
            resolution = (int(resolution[0]), int(resolution[1]))
        padded_images = np.zeros(images.shape[:1] + resolution, dtype=np.uint8)
        origin = ((resolution[0] - images.shape[1]) // 2, (resolution[1] - images.shape[2]) // 2)
        padded_images[:, padding[0]:padding[0] + images.shape[1], padding[2]:padding[2] + images.shape[2]] = images

        wf = Wavefront(wavelength, pixel_pitch, resolution, depth=images.shape[0])
        if intensity:
            wf.intensity = torch.tensor(padded_images * scale_intensity).double()
            wf.amplitude /= wf.amplitude.amax(dim=(2, 3), keepdim=True)
        else:
            wf.phase = torch.tensor(padded_images)
        return wf

    @property
    def shape(self):
        return (self.batch, self.depth,) + self.resolution

    @shape.setter
    def shape(self, shape):
        self.batch = shape[0]
        self.depth = shape[1]
        self.resolution = shape[:2]

    @property
    def amplitude(self):
        return self.u.abs()

    @property
    def scaled_amplitude(self):
        return self.amplitude / self.amplitude.amax(dim=(2,3), keepdim=True)

    @amplitude.setter
    def amplitude(self, new_amplitude):
        if len(new_amplitude.shape) < 4:
            new_amplitude = new_amplitude.reshape(self.shape)
        self.u = torch.polar(new_amplitude, self.phase)

    @property
    def phase(self):
        return self.u.angle()

    @phase.setter
    def phase(self, new_phase):
        if len(new_phase.shape) < 4:
            new_phase = new_phase.reshape(self.shape)
        self.u = torch.polar(self.amplitude, new_phase.reshape(self.shape))

    @property
    def intensity(self):
        return Wavefront.to_numpy(torch.square(self.amplitude))

    @intensity.setter
    def intensity(self, new_intensity):
        self.amplitude = torch.sqrt(new_intensity)

    @property
    def total_intensity(self):
        return Wavefront.to_numpy(torch.sum(torch.square(self.amplitude), (-2,-1)))

    @classmethod
    def to_numpy(cls, tt):
        return tt.squeeze(0).cpu().detach().numpy()

    def polar_to_rect(self, amp, phase):
        """from neural holo"""
        self.u = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase)).reshape(self.shape)

    def assert_equal(self, other_field, atol=1e-6):
        return torch.allclose(self.u, other_field.u, atol=atol)

    def plot(self, **kwargs):
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

        assert len(img.shape) == 3, "cannot plot intensity of batch of wavefronts"
        for i in range(self.depth):
            if options["threshold_foreground"]:
                img[i] = (img[i] > filters.threshold_otsu(img[i]))

            plt.imshow(img[i], cmap='gray')
            plt.xticks([]), plt.yticks([])
            if options['save']: plt.savefig(options['path'] + options['title'] + str(i+1) + '.jpg', bbox_inches="tight", pad_inches = 0)
            plt.colorbar()
            plt.title(options['title'] + '_plane_' + str(i+1))
            plt.show()
            plt.close()

    def copy(self, copy_wf=False):
        return copy.deepcopy(self) if copy_wf else Wavefront(self.wavelength, self.pixel_pitch, self.resolution, depth=self.depth, batch=self.batch)

    def requires_grad(self):
        self.u.requires_grad_(True)