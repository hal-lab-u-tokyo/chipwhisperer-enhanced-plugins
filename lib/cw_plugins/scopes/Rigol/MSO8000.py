
from cw_plugins.scopes.base import ScopeBase,TriggerMode
import time
import numpy as np
from pyvisa import VisaIOError

class MSO8000(ScopeBase):
    """
        Rigol MSO8000 series oscilloscope class
    """
    slope_str = {
        TriggerMode.EDGE_RISE: "POSitive",
        TriggerMode.EDGE_FALL: "POSitive",
        TriggerMode.EDGE_ANY: "RFALl"
    }

    mem_depth_options = [
        # 1k, 10k, 100k, 1M, 10M, 25M, 50M, 100M, 125M, 250M, 500M
        # in 2ch mode, upper limit is 250M
        # in 4ch mode, upper limit is 125M
        1e+3, 10e+3, 100e+3, 1e+6, 10e+6, 25e+6, 50e+6, 100e+6, 125e+6, 250e+6, 500e+6
    ]
    CHUNK_SIZE = 1000
    def __init__(self, model, resource, timeout):
        super().__init__(resource, timeout)
        self.max_sampling_rate = 5e+9 # 5Gsa/s
        self.max_mem_depth = 500e+6 # 500Mpts
        self.start_pos = 1
        self.sample_count = 1

        self.write(":RUN")
        # all channel display off
        for i in range(1, self.num_channels + 1):
            self.write(f":CHANnel{i}:DISPlay OFF")
        # init memory depth
        self.write(f":ACQuire:MDEPth 10k")
        # self.write(f':WAVEFORM:POINTS {self.CHUNK_SIZE:d}')
        time.sleep(0.2)
        # initialize with maximum sampling rate
        self.set_sampling_rate(self.max_sampling_rate)


    def get_num_channels(self):
        return 4

    def get_sampling_rate(self):
        return float(self.query(":ACQuire:SRATe?"))

    def set_sampling_rate(self, rate):
        if rate <= 0 or rate > self.max_sampling_rate:
            raise ValueError(f"Sampling rate {rate} is out of range")

        wave_length = self.__get_mem_depth() / rate / 2

        # set time scale
        self.write(f":TIMebase:SCALe {wave_length/10:e}") # 10 div

        time.sleep(0.5)
        # verify
        self.__verify_sampling_rate(rate)

    def __verify_sampling_rate(self, rate):
        if rate != self.sampling_rate:
            raise RuntimeError(f"Failed to set sampling rate to {rate} (actual {self.sampling_rate}))")

    def __set_mem_depth(self, depth):
        # if mem depth increased, time scale should also be increased to keep the sample rate

        if depth in self.mem_depth_options:

            current_srate = self.get_sampling_rate()
            scale = depth / self.__get_mem_depth()
            self.write(f":ACQuire:MDEPth {int(depth)}")

            # adjust time scale to keep the sampling rate
            current_scale = float(self.query(":TIMebase:SCALe?"))
            self.write(f":TIMebase:SCALe {current_scale * scale:e}")
            time.sleep(0.5)

            # check if sampling rate is the same
            self.__verify_sampling_rate(current_srate)
        else:
            raise ValueError(f"Memory depth {depth} is not supported")

    def __get_mem_depth(self):
        return float(self.query(":ACQuire:MDEPth?"))

    def set_vertical_scale(self, channel, scale):
        if not (1 <= channel <= self.num_channels):
            raise ValueError(f"Channel {channel} is out of range")
        self.write(f":CHANNEL{channel}:SCALE {scale}")

    def set_vertical_offset(self, channel, offset):
        if not (1 <= channel <= self.num_channels):
            raise ValueError(f"Channel {channel} is out of range")
        self.write(f":CHANNEL{channel}:OFFSET {offset}")


    def config_trigger_channel(self, mode, channel, scale, offset):
        if not (1 <= channel <= self.num_channels):
            raise ValueError(f"Channel {channel} is out of range")
        self.write(f":CHANnel{channel}:DISPlay ON")
        time.sleep(0.2)

        self.set_vertical_scale(channel, scale)
        self.set_vertical_offset(channel, offset)
        self.write(":TRIGger:MODE EDGE")
        self.write(f":TRIGger:EDGE:LEVel {scale:e}")
        self.write(f":TRIGger:EDGE:SOURCE CHAN{channel}")
        self.write(f":TRIGger:EDGE:SLOPe {self.slope_str[mode]}")


    def config_trace_channel(self, channel, scale, offset, period, delay = 0, impedance = None):

        if not (1 <= channel <= self.num_channels):
            raise ValueError(f"Channel {channel} is out of range")

        self.write(f":CHANnel{channel}:DISPlay ON")
        time.sleep(0.2)

        self.set_vertical_scale(channel, scale)
        self.set_vertical_offset(channel, offset)

        srate = self.sampling_rate

        points = (period + delay) * srate

        if points > self.__get_mem_depth():
            # increase memory depth
            for depth in self.mem_depth_options:
                if points <= depth:
                    break
            else:
                raise ValueError(f"Memory depth {depth} is not enough")
            self.__set_mem_depth(depth)

        self.sample_count = int(period * srate)

        if not impedance is None:
            if impedance == 50:
                impedance_str = "FIFTy"
            elif impedance == 1e6:
                impedance_str = "OMEG"

            self.write(f":CHANnel{channel}:IMPedance {impedance_str}")


        self.write(f":WAVEFORM:SOURCE CHANnel{channel}")
        self.write(':WAVEFORM:MODE RAW')
        self.write(":WAVEFORM:FORMAT BYTE")

        x_origin = float(self.query(':WAVEFORM:XORIGIN?'))
        self.start_pos = max(int((delay - x_origin) * srate ), 1)


    def is_triggered(self):
        return self.query(":TRIGGER:STATUS?").strip() == "STOP"

    def arm(self):
        self.write(":SINGLE")
        time.sleep(0.8)

    def get_last_trace(self, as_int):
        # read waveform
        waveform_data = np.array([], np.uint8)

        self.write(f':WAVEFORM:START {self.start_pos}')
        stop_pos = self.start_pos + self.sample_count
        self.write(f':WAVEFORM:STOP {stop_pos}')
        try:
            raw_data = self.resource.query_binary_values(':WAVEFORM:DATA?', datatype='B', container=np.array)
        except VisaIOError:
            return []
        
        waveform_data = np.append(waveform_data, raw_data)

        if as_int:
            return waveform_data
        else:
            try:
                y_increment = float(self.query(':WAVEFORM:YINCREMENT?'))
                y_origin = float(self.query(':WAVEFORM:YORIGIN?'))
                y_reference = int(self.query(':WAVEFORM:YREFERENCE?'))
            except ValueError:
                return []

            return ((waveform_data) * y_increment) + y_origin - y_increment * y_reference
