import cv2
import torch
import matplotlib.pyplot as plt

from optogenetic_holography.wavefield import Wavefield


def load_image(path, img_name, scale_intensity=1):
    img = cv2.imread(path + img_name, 0)
    return torch.tensor(img * scale_intensity, dtype=torch.complex128)


def save_image(wavefield: Wavefield, path, title="", plot=False):
    #plt.imshow(wavefield.intensity, cmap='gray', vmin=0, vmax=1)  # leads different result
    plt.imshow(wavefield.intensity, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.colorbar()
    plt.title(title)
    plt.savefig(path + 'img_{}.png'.format(title))
    if plot:
        plt.show()
    plt.close()
