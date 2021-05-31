"""deprecated"""

from abc import ABC, abstractmethod


class WavefrontInterface(ABC):

    @property
    @abstractmethod
    def wavelength(self):
        pass

    @property
    @abstractmethod
    def pixel_pitch(self):
        pass

    @property
    @abstractmethod
    def resolution(self):
        pass

    @property
    @abstractmethod
    def amplitude(self):
        pass

    @amplitude.setter
    def amplitude(self, new_amplitude):
        pass

    @property
    @abstractmethod
    def phase(self):
        pass

    @phase.setter
    def phase(self, phase):
        pass

    @property
    @abstractmethod
    def intensity(self):
        pass

    @property
    @abstractmethod
    def total_intensity(self):
        pass
