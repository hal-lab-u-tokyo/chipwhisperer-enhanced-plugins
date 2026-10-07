from abc import ABCMeta, abstractmethod
from enum import Enum
import math
from numbers import Real
import re
import time
from pyvisa import VisaIOError

class TriggerMode(Enum):
    EDGE_RISE = 0
    EDGE_FALL = 1
    EDGE_ANY = 2

# Abstract class for all oscilloscope classes
class ScopeBase(metaclass=ABCMeta):
    @staticmethod
    def decode_time(value, *, allow_percent=False):
        """Decode a capture duration or delay into ``(value, unit)``.

        Real numbers mean seconds. Strings require one of ``s``, ``ms``,
        ``us`` (also ``µs`` or ``μs``), ``ns``, ``ps``, or ``samples``.
        Time values are normalized to seconds; sample counts remain integers.
        With allow_percent=True, ``%`` is accepted and kept in percent units
        (``"-10%"`` returns ``(-10.0, "percent")``).

        Units are case-sensitive. Signs, scientific notation, and whitespace
        around the value and between the number and unit are accepted.
        No sampling-rate conversion or duration/delay range checking is done
        here: callers retain the unit until acquisition settings are known.
        """
        if isinstance(value, bool):
            raise TypeError("Time specification must be a real number or string")
        if isinstance(value, Real):
            number, unit = float(value), "s"
        elif isinstance(value, str):
            match = re.fullmatch(
                r"\s*([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
                r"(?:[eE][+-]?[0-9]+)?)\s*(s|ms|us|µs|μs|ns|ps|samples|%)\s*",
                value,
            )
            if match is None:
                raise ValueError(f"Invalid time specification: {value!r}")
            number, unit = float(match[1]), match[2]
        else:
            raise TypeError("Time specification must be a real number or string")

        if not math.isfinite(number):
            raise ValueError("Time specification must be finite")
        if unit == "samples":
            if not number.is_integer():
                raise ValueError("Sample count must be an integer")
            return int(number), "samples"
        if unit == "%":
            if not allow_percent:
                raise ValueError("Percent is only supported for delay")
            return number, "percent"
        factors = {"s": 1, "ms": 1e-3, "us": 1e-6, "µs": 1e-6,
                   "μs": 1e-6, "ns": 1e-9, "ps": 1e-12}
        return number * factors[unit], "seconds"

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
