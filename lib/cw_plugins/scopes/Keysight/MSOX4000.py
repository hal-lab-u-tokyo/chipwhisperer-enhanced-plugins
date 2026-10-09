
from cw_plugins.scopes.base import ScopeBase,TriggerMode
import time
import numpy as np

# import warnings
from logging import warning

class MSOX4000(ScopeBase):
    """
        Keysight InfiniiVision 4000 X series oscilloscope class
    """
    specs = { # sampling rate, channel, mem depth
        "MSO-X 4104A": (5e9, 4, 4e6)
    }
    slope_str = {
        TriggerMode.EDGE_RISE: "POSitive",
        TriggerMode.EDGE_FALL: "NEGative",
        TriggerMode.EDGE_ANY: "EITHer"
    }
    time_range_list = [8e-4, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, \
                       1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7, 5e-8, 2e-8, 1e-8, 5e-9]
    NORMAL_MODE_MAX_SAMPLE = 62500

    def __init__(self, model, resource, timeout):
        super().__init__(resource, timeout)
        self.__model = model
        self.max_sampling_rate = self.specs[self.__model][0]
        self.max_mem_depth = self.specs[self.__model][2]
        self.skip_samples = 0
        self.capture_samples = 1

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
        rate = self.decode_sampling_rate(rate)
        if rate <= 0 or rate > self.max_sampling_rate:
            raise ValueError(f"Sampling rate {rate} is out of range")
        self.write(f":ACQuire:SRATe {rate:e}")
        time.sleep(0.5)
        # verify
        if rate != self.sampling_rate:
            raise RuntimeError(f"Failed to set sampling rate to {rate} (actual {self.sampling_rate}))")

    def adjust_time_range(self):
        # set suitable time range
        rate = self.sampling_rate
        for i in range(len(self.time_range_list)):
            if int(self.time_range_list[i] * rate) <= self.NORMAL_MODE_MAX_SAMPLE:
                break
        self.write(f":TIMEBASE:RANGE {self.time_range_list[i]}")
        return self.time_range_list[i]

    def set_vertical_scale(self, channel, scale):
        if not (1 <= channel <= self.num_channels):
            raise ValueError(f"Channel {channel} is out of range")
        scale = self.decode_voltage(scale)
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.write(f":CHANnel{channel}:SCALe {scale:e}")

    def set_vertical_offset(self, channel, offset):
        if not (1 <= channel <= self.num_channels):
            raise ValueError(f"Channel {channel} is out of range")
        offset = self.decode_voltage(offset)
        self.write(f":CHANnel{channel}:OFFSet {offset:e}")


    def config_trigger_channel(self, mode, channel, scale, offset, threshold=None):
        """Configure an edge trigger with a GND-referenced threshold in V.

        scale (V/div), offset (V), and threshold (V) accept unit strings.
        None selects one division above the range center: offset + scale.
        """
        if not (1 <= channel <= self.num_channels):
            raise ValueError(f"Channel {channel} is out of range")
        scale = self.decode_voltage(scale)
        offset = self.decode_voltage(offset)
        threshold = self.decode_voltage(offset + scale if threshold is None else threshold)
        if scale <= 0:
            raise ValueError("scale must be positive")
        slope = self.slope_str[mode]
        self.write(f":CHANnel{channel}:DISPlay ON")
        self.set_vertical_scale(channel, scale)
        self.set_vertical_offset(channel, offset)
        self.write(":TRIGger:MODE EDGE")
        self.write(f":TRIGger:EDGE:SOURce CHANnel{channel}")
        self.write(f":TRIGger:EDGE:SLOPE {slope}")
        self.write(f":TRIGger:EDGE:LEVel {threshold:e}")

    def set_close_points(self, points):
        set_points = points
        history = []
        while set_points <= int(self.max_mem_depth):
            self.write(f":WAVEFORM:POINTS {set_points}")
            time.sleep(0.2)
            current_points = int(self.query(":WAVEFORM:POINTS?"))
            history.append(current_points)
            # max trial to avoid infinite loop
            if history.count(current_points) > 10:
                break
            elif current_points < points:
                diff = points - current_points
                set_points += diff
                set_points = min(set_points, int(self.max_mem_depth))
                prev_points = current_points
            else:
                break

        return current_points

    def config_trace_channel(self, channel, scale, offset, period, delay = 0, impedance = None):
        """Configure trace duration and start relative to the trigger.

        period/delay accept seconds or time/sample-count strings. delay also
        accepts a signed percentage of period (negative means pre-trigger).
        Sample counts are converted using the current sampling rate.
        """

        if not (1 <= channel <= self.num_channels):
            raise ValueError(f"Channel {channel} is out of range")

        scale = self.decode_voltage(scale)
        offset = self.decode_voltage(offset)
        duration, duration_unit = self.decode_time(period)
        start, start_unit = self.decode_time(delay, allow_percent=True)
        if scale <= 0:
            raise ValueError("scale must be positive")
        if duration <= 0:
            raise ValueError("period must be positive")
        rate = self.sampling_rate
        period = duration / rate if duration_unit == "samples" else duration
        if start_unit == "samples":
            delay = start / rate
        elif start_unit == "percent":
            delay = period * start / 100
        else:
            delay = start
        ranges = [value for value in self.time_range_list if value >= period * 2]
        if not ranges:
            raise ValueError("period exceeds the supported time range")
        if round(period * rate) < 1:
            raise ValueError("period must span at least one sample")

        # first, set max mem depth
        self.write(f":ACQuire:POINts {self.max_mem_depth}")
        # set max time range in normal mode
        time_range = self.adjust_time_range()

        # if  period * 2 > time_range:
            # change to RAW mode
        mode = "RAW"
        # find suitable time range
        new_time_range = ranges[-1]
        self.write(f":TIMEBASE:RANGE {new_time_range}")
        desired_points = int(new_time_range * self.sampling_rate)
        # else:
        #     mode = "NORMAL"
        #     desired_points = int(period * self.sampling_rate * 2)

        self.set_vertical_scale(channel, scale)
        self.set_vertical_offset(channel, offset)

        self.write(":ACQuire:DIGitizer ON")
        self.write(f":WAVEFORM:SOURCE CHANnel{channel}")
        self.write(":WAVEFORM:FORMAT WORD")
        self.write(f":TIMEBASE:DELAY {delay:e}")
        self.write(f":WAVEFORM:POINTS:MODE {mode}")

        points = self.set_close_points(desired_points)

        if points < desired_points:
            warning(f"Failed to set points to {desired_points} (actual {points})")

        # if mode == "NORMAL":
        #     self.write(f":TIMEBASE:RANGE {points / self.sampling_rate}")

        self.capture_samples = (duration if duration_unit == "samples"
                                else round(period * self.sampling_rate))

        if not impedance is None:
            if impedance == 50:
                impedance_str = "FIFTy"
            elif impedance == 1e6:
                impedance_str = "ONEMeg"

            self.write(f":CHANnel{channel}:IMPedance {impedance_str}")

        self.write(f":CHANnel{channel}:DISPlay ON")

    def is_triggered(self):
        return int(self.query(":TER?"))

    def arm(self):
        # clear status
        self.write("*CLS")
        self.write(":SINGLE")
        time.sleep(0.1)


    def get_last_trace(self, as_int):
        # read waveform
        wavedata = self.resource.query_binary_values(":WAVEFORM:DATA?", datatype="H", container=np.array)

        start_pos = len(wavedata) // 2
        wavedata = wavedata[start_pos:start_pos+self.capture_samples]

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
