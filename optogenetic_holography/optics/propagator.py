from abc import ABC, abstractmethod
#from optogenetic_holography.optics.wavefront import WavefrontInterface


class Propagator(ABC):

    @abstractmethod
    def forward(self, wf):
        pass

    @abstractmethod
    def backward(self, wf):
        pass
