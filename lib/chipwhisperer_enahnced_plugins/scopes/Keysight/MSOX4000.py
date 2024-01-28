
from sca_tools.scopes.base import ScopeBase,TriggerMode
import time
import numpy as np

class MSOX4000(ScopeBase):
    """
        Keysight InfiniiVision 4000 X series oscilloscope class
    """
    specs = { # sampling rate, channel, mem depth
        "MSO-X 4104A": (5e9, 4, 4e6)
    }
    slope_str = {
        TriggerMode.EDGE_RASE: "POSitive",
        TriggerMode.EDGE_FALL: "NEGative",
        TriggerMode.EDGE_ANY: "EITHer"
    }
    def __init__(self, model, resource, timeout):
        super().__init__(resource, timeout)
        self.__model = model
        self.max_sampling_rate = self.specs[self.__model][0]
        self.max_mem_depth = self.specs[self.__model][2]
        self.skip_samples = 0

        self.write(":RUN")
        # all channel display off
        for i in range(1, self.num_channels + 1):
            self.write(f":CHANnel{i}:DISPlay OFF")
        time.sleep(0.2)
        # initialize with maximum sampling rate
        self.set_sampling_rate(self.max_sampling_rate)


    def get_num_channels(self):
        return self.specs[self.__model][1]

    def get_sampling_rate(self):
        return float(self.query(":ACQuire:SRATe?"))

    def set_sampling_rate(self, rate):
        if rate <= 0 or rate > self.max_sampling_rate:
            raise ValueError(f"Sampling rate {rate} is out of range")
        self.write(f":ACQuire:SRATe {rate:e}")
        time.sleep(0.5)
        # verify
        if rate != self.sampling_rate:
            raise RuntimeError(f"Failed to set sampling rate to {rate} (actual {self.sampling_rate}))")

    def set_vertical_scale(self, channel, scale):
        if not (1 <= channel <= self.num_channels):
            raise ValueError(f"Channel {channel} is out of range")
        self.write(f":CHANnel{channel}:SCALe {scale:e}")

    def set_vertical_offset(self, channel, offset):
        if not (1 <= channel <= self.num_channels):
            raise ValueError(f"Channel {channel} is out of range")
        self.write(f":CHANnel{channel}:OFFSet {offset:e}")


    def config_trigger_channel(self, mode, channel, scale, offset):
        if not (1 <= channel <= self.num_channels):
            raise ValueError(f"Channel {channel} is out of range")
        self.write(":TRIGger:MODE EDGE")
        self.write(":TRIGger:EDGE:LEVel 1")
        self.write(f":TRIGger:EDGE:SOURce CHANnel{channel}")
        self.write(f":TRIGger:EDGE:SLOPE {self.slope_str[mode]}")
        self.write(f":CHANnel{channel}:DISPlay ON")
        self.set_vertical_scale(channel, scale)
        self.set_vertical_offset(channel, offset)

    def config_trace_channel(self, channel, scale, offset, period, delay = 0, impedance = None):

        if not (1 <= channel <= self.num_channels):
            raise ValueError(f"Channel {channel} is out of range")

        self.set_vertical_scale(channel, scale)
        self.set_vertical_offset(channel, offset)

        # calculate number of points from length
        # tigger point is in the middle of the trace so multiplied by 2
        capture_range = (period + delay) * 2
        points = int(capture_range * self.sampling_rate)
        self.skip_samples = int(delay * self.sampling_rate)

        if points > self.max_mem_depth // 2:
            Warning(f"Trace period {period} is too long, set to {self.max_mem_depth / self.sampling_rate / 2}")
            points = int(self.max_mem_depth // 2)

        if not impedance is None:
            if impedance == 50:
                impedance_str = "FIFTy"
            elif impedance == 1e6:
                impedance_str = "ONEMeg"

            self.write(f":CHANnel{channel}:IMPedance {impedance_str}")

        self.write(f":timebase:range {capture_range:e}")


        self.write(f":WAVEFORM:SOURCE CHANnel{channel}")
        self.write(":WAVEFORM:POINTS:MODE NORMAL")
        self.write(":WAVEFORM:FORMAT BYTE")

        self.write(f":WAVEFORM:POINTS {points:d}")
        self.write(f":CHANel{channel}:DISPlay ON")

    def is_triggered(self):
        return int(self.query(":TER?"))

    def arm(self):
        # clear status
        self.write("*CLS")
        self.write(":SINGLE")
        time.sleep(0.1)


    def get_last_trace(self, as_int):
        # read waveform
        wavedata = self.resource.query_binary_values(":WAVEFORM:DATA?", datatype="B", container=np.array)

        start_pos = len(wavedata) // 2 + self.skip_samples
        wavedata = wavedata[start_pos:]

        if as_int:
            return wavedata
        else:
            try:
                y_increment = float(self.query(':WAVEFORM:YINCREMENT?'))
                y_origin = float(self.query(':WAVEFORM:YORIGIN?'))
                y_reference = int(self.query(':WAVEFORM:YREFERENCE?'))
            except ValueError:
                return None

            return ((wavedata) * y_increment) + y_origin - y_increment * y_reference