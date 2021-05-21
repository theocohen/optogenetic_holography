from abc import ABC, abstractmethod
#from optogenetic_holography.optics.wavefront import WavefrontInterface


class Propagator(ABC):

    @abstractmethod
    def forward(self, wf, z):
        pass

    @abstractmethod
    def backward(self, wf, z):
        pass
