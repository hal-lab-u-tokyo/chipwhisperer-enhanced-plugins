from abc import ABCMeta, abstractmethod
from enum import Enum
import time
from pyvisa import VisaIOError

class TriggerMode(Enum):
    EDGE_RASE = 0
    EDGE_FALL = 1
    EDGE_ANY = 2

# Abstract class for all oscilloscope classes
class ScopeBase(metaclass=ABCMeta):
    def __init__(self, resource, timeout):
        """
            resource: pyvisa resource
            wait_time: wait time after arming
            timeout: timeout in ms
        """
        self.resource = resource
        self.timeout = timeout
        self.resource.timeout = 3000 # 3s

    # destractor
    def __del__(self):
        self.resource.close()

    def write(self, cmd):
        try:
            self.resource.write(cmd)
        except VisaIOError:
            return False
        return True

    def query(self, cmd, default = ""):
        """
            wrapper for pyvisa query

            Args:
                cmd: command to send
                default: default value to return if the command fails
        """
        try:
            return self.resource.query(cmd)
        except VisaIOError:
            return default

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
    def config_trace_channel(self, channel, scale, offset, period, delay = 0, **kwargs):
        """
            channel: channel number
            scale: vertical scale in V
            offset: vertical offset in V
            period: time period in s
            delay: delay time to acquisition after trigger in s
            **kwargs: oscilloscope dependent parameters
        """
        pass

    @abstractmethod
    def is_triggered(self):
        """
            Return whether the oscilloscope is triggered or not
        """
        pass

    # ChipWhisperer compatible interface
    @abstractmethod
    def arm(self):
        """Setup scope for triggering"""
        pass

    def capture(self, **kwargs) -> bool:
        time.sleep(0.2)
        # wait untill the scope is triggered
        start = time.time()
        while not self.is_triggered():
            if time.time() - start > self.timeout:
                return True
            time.sleep(0.5)
        return False


    @abstractmethod
    def get_last_trace(self, as_int):
        """Return the captured waveform"""
        pass
