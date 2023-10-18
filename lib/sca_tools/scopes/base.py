from abc import ABCMeta, abstractmethod
from enum import Enum

class TriggerMode(Enum):
    EDGE_RASE = 0
    EDGE_FALL = 1
    EDGE_ANY = 2

# Abstract class for all oscilloscope classes
class ScopeBase(metaclass=ABCMeta):
    def __init__(self, resource, timeout):
        """
            resource: pyvisa resource
            timeout: timeout in ms
        """
        self.resource = resource
        self.timeout = timeout
        self.resource.timeout = self.timeout

    # destractor
    def __del__(self):
        self.resource.close()

    def write(self, cmd):
        self.resource.write(cmd)

    def query(self, cmd):
        return self.resource.query(cmd)

    @abstractmethod
    def get_num_channels(self):
        """
            Number of channels of the oscilloscope
        """
        pass

    @property
    def num_channels(self):
        return self.get_num_channels()

    @abstractmethod
    def get_sampling_rate(self):
        """
            Sampling rate of the oscilloscope
        """
        pass

    @abstractmethod
    def set_sampling_rate(self, rate):
        pass

    @property
    def sampling_rate(self):
        return self.get_sampling_rate()

    @sampling_rate.setter
    def sampling_rate(self, rate):
        self.set_sampling_rate(rate)

    @abstractmethod
    def config_trigger_channel(self, mode, channel, scale, offset):
        """
            mode: TriggerMode
            channel: channel number
            scale: vertical scale in V
            offset: vertical offset in V
        """
        pass

    @abstractmethod
    def config_trace_channel(self, channel, scale, offset, period, impedance = None):
        """
            channel: channel number
            scale: vertical scale in V
            offset: vertical offset in V
            period: time period in s
            impedance: impedance in ohm
        """
        pass

    @abstractmethod
    def capture(self, target):
        pass

