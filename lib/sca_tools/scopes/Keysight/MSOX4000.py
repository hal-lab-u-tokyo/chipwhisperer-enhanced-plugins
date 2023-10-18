
from sca_tools.scopes.base import ScopeBase,TriggerMode
import time
import asyncio
import numpy as np

class MSOX4000(ScopeBase):
    """
        Keysight InfiniiVision 4000 X series oscilloscope class
    """
    specs = { # sampling rate, channel
        "MSO-X 4104A": (5e9, 4)
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
        self.x_increment = 1
        self.x_origin = 0
        self.x_reference = 0
        self.y_increment = 1
        self.y_origin = 0
        self.y_reference = 0
        # all channel display off
        for i in range(1, self.num_channels + 1):
            self.write(f":CHANnel{i}:DISPlay OFF")
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
        time.sleep(0.2)
        # verify
        if rate != self.sampling_rate:
            raise RuntimeError(f"Failed to set sampling rate to {rate}")

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

    def config_trace_channel(self, channel, scale, offset, period, impedance = None):
        if not (1 <= channel <= self.num_channels):
            raise ValueError(f"Channel {channel} is out of range")
        self.set_vertical_scale(channel, scale)
        self.set_vertical_offset(channel, offset)
        # calculate number of points from length
        points = int(period * self.sampling_rate)
        self.write(f":WAVEFORM:SOURCE CHANnel{channel}")
        self.write(":WAVEFORM:FORMAT BYTE")
        self.write(":WAVEFORM:POINTS:MODE MAXIMUM")
        self.write(f":WAVEFORM:POINTS {points:d}")
        self.write(":acquire:digitizer 1")
        self.write(f":acquire:points {points:d}")

        if not impedance is None:
            if impedance == 50:
                impedance_str = "FIFTy"
            elif impedance == 1e6:
                impedance_str = "ONEMeg"

            self.write(f":CHANnel{channel}:IMPedance {impedance_str}")

        self.write(f":CHANel{channel}:DISPlay ON")

        time.sleep(0.2)
        self.x_increment = float(self.query(':WAVEFORM:XINCREMENT?'))
        self.x_origin = float(self.query(':WAVEFORM:XORIGIN?'))
        self.x_reference = int(self.query(':WAVEFORM:XREFERENCE?'))

        self.y_increment = float(self.query(':WAVEFORM:YINCREMENT?'))
        self.y_origin = float(self.query(':WAVEFORM:YORIGIN?'))
        self.y_reference = int(self.query(':WAVEFORM:YREFERENCE?'))
        print(self.x_increment,self.x_origin,self.x_reference,self.y_increment,self.y_origin,self.y_reference)
        # clear status
        self.write("*CLS")

    async def wait_for_trigger(self):
        self.write(":SINGLE")
        while int(self.query(":TER?")) == 0:
            await asyncio.sleep(0.1)

    def capture(self, target):
        # run_target()
        coro = asyncio.wait_for(self.wait_for_trigger(), 10)
        try:
            asyncio.run(coro)
        except asyncio.TimeoutError:
            raise RuntimeError("Timeout while waiting for trigger")

        # read waveform
        wavedata = self.resource.query_binary_values(":WAVEFORM:DATA?", datatype="B", container=np.array)

        return ((wavedata) * self.y_increment) + self.y_origin - self.y_increment * self.y_reference