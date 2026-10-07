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
    def _decode_quantity(value, units, default_unit, name):
        """Parse a finite real number and a case-sensitive unit."""
        if isinstance(value, bool):
            raise TypeError(f"{name} specification must be a real number or string")
        if isinstance(value, Real):
            number, unit = float(value), default_unit
        elif isinstance(value, str):
            unit_pattern = "|".join(re.escape(unit) for unit in units)
            match = re.fullmatch(
                r"\s*([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
                r"(?:[eE][+-]?[0-9]+)?)\s*(" + unit_pattern + r")\s*",
                value,
            )
            if match is None:
                raise ValueError(f"Invalid {name.lower()} specification: {value!r}")
            number, unit = float(match[1]), match[2]
        else:
            raise TypeError(f"{name} specification must be a real number or string")
        if not math.isfinite(number):
            raise ValueError(f"{name} specification must be finite")
        return number, unit

    @staticmethod
    def _decode_scaled_quantity(value, factors, default_unit, name):
        number, unit = ScopeBase._decode_quantity(value, factors, default_unit, name)
        result = number * factors[unit]
        if not math.isfinite(result):
            raise ValueError(f"{name} specification must be finite")
        return result

    @staticmethod
    def decode_voltage(value):
        """Return volts from a number or V/mV/uV/µV/μV/nV/kV string.

        Units are case-sensitive. Signs, scientific notation, and whitespace
        are accepted. Hardware limits are checked by callers.
        """
        factors = {"V": 1, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6,
                   "μV": 1e-6, "nV": 1e-9, "kV": 1e3}
        return ScopeBase._decode_scaled_quantity(value, factors, "V", "Voltage")

    @staticmethod
    def decode_sampling_rate(value):
        """Return samples/s from a positive number or unit string.

        Accept S/s/kS/s/MS/s/GS/s and
        SPS/kSPS/MSPS/GSPS. Units are case-sensitive.
        """
        factors = {"S/s": 1, "kS/s": 1e3, "MS/s": 1e6, "GS/s": 1e9,
                   "SPS": 1, "kSPS": 1e3, "MSPS": 1e6, "GSPS": 1e9}
        rate = ScopeBase._decode_scaled_quantity(value, factors, "S/s", "Sampling rate")
        if rate <= 0:
            raise ValueError("Sampling rate must be positive")
        return rate

    @staticmethod
    def decode_time(value, *, allow_percent=False):
        """Return (value, unit), retaining samples and percent units.

        Numbers mean seconds. Strings require s/ms/us/µs/μs/ns/ps,
        samples, or (with allow_percent=True) %. Times normalize to seconds;
        sample counts must be integers. "-10%" returns (-10.0, "percent").
        Signs, scientific notation, and whitespace are accepted. No rate
        conversion or duration/delay range checking is performed here.
        """
        factors = {"s": 1, "ms": 1e-3, "us": 1e-6, "µs": 1e-6,
                   "μs": 1e-6, "ns": 1e-9, "ps": 1e-12}
        number, unit = ScopeBase._decode_quantity(
            value, (*factors, "samples", "%"), "s", "Time")
        if unit == "samples":
            if not number.is_integer():
                raise ValueError("Sample count must be an integer")
            return int(number), "samples"
        if unit == "%":
            if not allow_percent:
                raise ValueError("Percent is only supported for delay")
            return number, "percent"
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
    def config_trigger_channel(self, mode, channel, scale, offset, threshold=None, **kwargs):
        """
            mode: TriggerMode
            channel: channel number
            scale: vertical scale in V
            offset: vertical offset in V
            threshold: trigger threshold in V; None selects offset + scale
            **kwargs: oscilloscope dependent parameters
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
