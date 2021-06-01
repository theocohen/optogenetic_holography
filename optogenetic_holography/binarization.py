import numpy as np
import torch
from skimage import filters
import mahotas

from optogenetic_holography.optics import optics_backend as opt


def from_phase_to_bin_amp(holo_wf):
    # fixme
    return (holo_wf.phase > 0).double()


def from_amp_to_bin_amp(holo_wf, method="mean"):
    """
    if method == "mean":
        return (holo_wf.amplitude > holo_wf.amplitude.mean()).double()
    elif method == "none":
        return holo_wf.amplitude
    """

    # img = (holo_wf.scaled_amplitude * 255).int().squeeze().numpy().astype(np.uint8)
    # threshold = mahotas.rc(img)
    threshold_method = None
    if method == "isodata":
        threshold_method = filters.threshold_isodata
    elif method == "otsu":
        threshold_method = filters.threshold_otsu
    elif method == "minimum":
        threshold_method = filters.threshold_minimum
    elif method == "mean":
        threshold_method = filters.threshold_mean
    elif method == "li":
        threshold_method = filters.threshold_li
    elif method == "niblack":
        threshold_method = filters.threshold_niblack
    elif method == "sauvola":
        threshold_method = filters.threshold_sauvola
    elif method == "triangle":
        threshold_method = filters.threshold_triangle
    elif method == "yen":
        threshold_method = filters.threshold_yen

    img_stack = opt.Wavefront.to_numpy(holo_wf.amplitude)
    thresholds = []
    for img in img_stack:
        thresholds.append(threshold_method(img))
    thresholds = torch.Tensor(thresholds).unsqueeze(1)
    return (holo_wf.amplitude > thresholds).double()
