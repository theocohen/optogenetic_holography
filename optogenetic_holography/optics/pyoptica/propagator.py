"""Adapter class of PyOptica library"""

import pyoptica as po
import logging

from optogenetic_holography.optics.propagator import Propagator
from optogenetic_holography.optics.pyoptica.wavefront import Wavefront


class FourierLensPropagator(Propagator):
    def __init__(self, radius, focal_length):
        self.lens = po.ThinLens(2 * radius, focal_length)
        self.fs_f = po.FreeSpace(focal_length)
        self.fs_b = po.FreeSpace(-focal_length)

    def forward(self, wf: Wavefront) -> Wavefront:
        wf_f1 = self.fs_f * wf.wf
        wf_lens = self.lens * wf_f1.wf
        propagated_wf = wf.copy()
        propagated_wf.wf = self.fs_f * wf_lens.wf
        return propagated_wf

    def backward(self, wf: Wavefront) -> Wavefront:
        wf_b = self.fs_b * wf.wf
        wf_lens = self.lens * wf_b.wf  # fixme not working as inverse
        propagated_wf = wf.copy()
        propagated_wf.wf = self.fs_b * wf_lens.wf
        return propagated_wf


class FresnelPropagator(Propagator):
    def __init__(self):
        self.kernels = {}

    def forward(self, wf: Wavefront, z) -> Wavefront:
        if z not in self.kernels:
            self.kernels[z] = po.FreeSpace(z)
        propagated_wf = wf.copy()
        propagated_wf.wf = self.kernels[z] * wf.wf
        return propagated_wf

    def backward(self, wf: Wavefront, z) -> Wavefront:
        return self.forward(wf, -z)


class RandomPhaseMask(Propagator):

    def forward(self, wf):
        diff = po.Diffuser()
        masked_wf = wf.copy()
        masked_wf.u = diff * wf.wf
        return masked_wf

    def backward(self, wf):
        pass
