# This code was made and tested based on 3418E.

from cw_plugins.scopes.base import ScopeBase, TriggerMode
import numpy as np
import pypicosdk as psdk

class PicoScope3000E(ScopeBase):
    """
        PicoScope 3000E series oscilloscope class
    """

    slope_map = {
        TriggerMode.EDGE_RISE: psdk.TRIGGER_DIR.RISING,
        TriggerMode.EDGE_FALL: psdk.TRIGGER_DIR.FALLING,
        TriggerMode.EDGE_ANY:  psdk.TRIGGER_DIR.RISING_OR_FALLING,
    }

    channel_map = {
        "A": psdk.CHANNEL.A,
        "B": psdk.CHANNEL.B,
        "C": psdk.CHANNEL.C,
        "D": psdk.CHANNEL.D,
    }

    coupling_map = {
        "AC": psdk.COUPLING.AC,
        "DC": psdk.COUPLING.DC,
    }

    def __init__(self, model="PicoScope3000E", resource=None, timeout=5000):
    # resource/timeout are kept for API compatibility with ScopeBase.
    # PicoScope does not use VISA resource, so do not call ScopeBase.__init__().
        self.resource = resource if resource is not None else DummyResource()
        self.timeout = timeout
        self._closed = False
    
        self.__model = model
        self.scope = psdk.psospa()
        self.scope.open_unit()

        # Default sampling rate. 
        # The actual PicoScope timebase is configured only after set_sampling_rate() is called.
        self._sampling_rate = 1.25e9
        self.timebase = None

        self.samples = 3000
        self.pre_trig_percent = 50

        self.trace_channel = psdk.CHANNEL.A
        self.trigger_channel = psdk.CHANNEL.B

        self.output_unit = "mv"
        self.time_unit = "ns"

        self.last_channel_buffer = None
        self.last_time_axis = None
        self.last_trace = None
        

    def close(self):
        if getattr(self, "_closed", False):
            return
        try:
            self.scope.close_unit()
        except Exception as e:
            print(f"PicoScope close ignored: {e}")
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_num_channels(self):
        #The second number of the model number represents the number of channels. e.g. 4 for 3418E.
        num = self.scope.get_unit_info(psdk.UNIT_INFO.PICO_VARIANT_INFO)
        return int(num[1])
    
    def is_triggered(self):
        """
        Not used for PicoScope block captures.
        Capture completion is handled internally by pypicosdk.get_values().
        """
        return 0

    def get_sampling_rate(self):
        return self._sampling_rate

    def set_sampling_rate(self, rate):
        if rate <= 0:
            raise ValueError(f"Sampling rate {rate} is invalid")

        self.requested_sampling_rate = rate

        self.timebase = self.scope.sample_rate_to_timebase(
            sample_rate=rate / 1e9,
            unit=psdk.SAMPLE_RATE.GSPS,
        )

        self._sampling_rate = self.scope.get_actual_sample_rate()
        print(
            f"PicoScope actual sampling rate set to "
            f"{self._sampling_rate/1e9:.3f} GS/s "
            f"(requested {rate/1e9:.3f} GS/s)"
        )
        return self._sampling_rate

    
    def config_trigger_channel(
        self,
        mode,
        channel,
        scale=None,
        offset=None,
        threshold_mv=50,
        coupling="DC",
        voltage_range=psdk.RANGE.mV500,
    ):
        if channel not in self.channel_map:
            raise ValueError(f"Channel {channel} is out of range")

        pico_ch = self.channel_map[channel]
        self.trigger_channel = pico_ch

        self.scope.set_channel(
            channel=pico_ch,
            coupling=self.coupling_map[coupling],
            range=voltage_range,
        )

        self.scope.set_simple_trigger(
            channel=pico_ch,
            threshold=threshold_mv,
            direction=self.slope_map[mode],
            delay=0,
        )

    def config_trace_channel(
        self,
        channel,
        scale=None,          # Unused (API compatibility)
        offset=None,         # Unused (API compatibility)
        period=None,
        delay=0,             # Unused (API compatibility)
        impedance=None,      # Unused (API compatibility)
        coupling="AC",
        voltage_range=psdk.RANGE.mV20,
        samples=None,
        pre_trig_percent=50,
    ):
        if channel not in self.channel_map:
            raise ValueError(f"Channel {channel} is out of range")

        pico_ch = self.channel_map[channel]
        self.trace_channel = pico_ch

        self.scope.set_channel(
            channel=pico_ch,
            coupling=self.coupling_map[coupling],
            range=voltage_range,
        )

        if pre_trig_percent < 0 or pre_trig_percent > 100:
            raise ValueError(f"pre_trig_percent {pre_trig_percent} is out of range")
        self.pre_trig_percent = pre_trig_percent

        if samples is not None:
            self.samples = int(samples)
        elif period is not None:
            self.samples = int(period * self._sampling_rate)
        else:
            raise ValueError("Either samples or period must be specified")

    def arm(self):
        if self.timebase is None:
            raise RuntimeError(
                "PicoScope timebase is not configured. "
                "Call set_sampling_rate() before arm()."
                )
        self.last_channel_buffer = self.scope.set_data_buffer_for_enabled_channels(
            self.samples
        )

        self.scope.run_block_capture(
            timebase=self.timebase,
            samples=self.samples,
            pre_trig_percent=self.pre_trig_percent,
            segment=0,
        )

        # time.sleep(0.001)

    def capture(self, poll_done=False):
        if self.last_channel_buffer is None:
            raise RuntimeError("PicoScope is not armed. Call arm() before capture().")
        try:
            actual_samples = self.scope.get_values(
                self.samples,
                start_index=0,
                segment=0,
                ratio=0,
                ratio_mode=psdk.RATIO_MODE.RAW,
            )

            for ch in self.last_channel_buffer:
                self.last_channel_buffer[ch] = self.last_channel_buffer[ch][:actual_samples]

            if self.output_unit == "mv":
                self.last_channel_buffer = self.scope.adc_to_mv(
                    self.last_channel_buffer
                )
            elif self.output_unit == "v":
                self.last_channel_buffer = self.scope.adc_to_volts(
                    self.last_channel_buffer
                )
            elif self.output_unit != "adc":
                raise ValueError(
                    f"Unsupported output unit: {self.output_unit}"
                )

            self.last_time_axis = self.scope.get_time_axis(
                self.timebase,
                actual_samples,
                pre_trig_percent=self.pre_trig_percent,
                ratio=0,
                unit=self.time_unit,
            )
            
            self.last_trace = self.last_channel_buffer[self.trace_channel]
            return False

        except Exception as e:
            print(
                f"PicoScope capture failed: {type(e).__name__}: {e} "
                f"(samples={self.samples}, timebase={self.timebase}, "
                f"trace_channel={self.trace_channel}, output_unit={self.output_unit})"
            )
            return True

    def get_last_trace(self, as_int=False):
        if self.last_trace is None:
            return None

        if as_int:
            return np.asarray(self.last_trace).astype(np.int16)

        return np.asarray(self.last_trace)

    def get_last_time_axis(self):
        return self.last_time_axis
    
class DummyResource:
    def close(self):
        pass