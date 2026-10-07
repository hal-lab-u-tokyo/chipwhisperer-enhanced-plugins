# This code was made and tested based on 3418E.

from cw_plugins.scopes.base import ScopeBase, TriggerMode
import math
from numbers import Real
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

    # Eight vertical divisions: the bipolar range peak spans four divisions.
    scale_map = {
        0.010 / 4: psdk.RANGE.mV10,
        0.020 / 4: psdk.RANGE.mV20,
        0.050 / 4: psdk.RANGE.mV50,
        0.100 / 4: psdk.RANGE.mV100,
        0.200 / 4: psdk.RANGE.mV200,
        0.500 / 4: psdk.RANGE.mV500,
        1.0 / 4: psdk.RANGE.V1,
        2.0 / 4: psdk.RANGE.V2,
        5.0 / 4: psdk.RANGE.V5,
        10.0 / 4: psdk.RANGE.V10,
        20.0 / 4: psdk.RANGE.V20,
    }

    def _channel_voltage_settings(self, scale, offset, voltage_range, default_range):
        if scale is not None:
            if isinstance(scale, bool) or not isinstance(scale, Real):
                raise ValueError(
                    f"Unsupported scale {scale!r}; supported V/div values: {list(self.scale_map)}"
                )
            if voltage_range is not None:
                raise ValueError("Specify either scale or voltage_range, not both")
            for supported_scale, supported_range in self.scale_map.items():
                if math.isclose(scale, supported_scale, rel_tol=1e-12, abs_tol=0):
                    voltage_range = supported_range
                    break
            else:
                raise ValueError(
                    f"Unsupported scale {scale!r}; supported V/div values: {list(self.scale_map)}"
                )
        elif voltage_range is None:
            voltage_range = default_range
        if offset is None:
            offset = 0.0
        if isinstance(offset, bool) or not isinstance(offset, Real) or not math.isfinite(offset):
            raise ValueError("offset must be a finite voltage in V")
        # The shared offset is the input voltage at the range center.
        # PicoSDK instead specifies the voltage added before digitization.
        return voltage_range, -float(offset)

    def __init__(self, resource=None, timeout=5000):
        # resource/timeout are kept for API compatibility with ScopeBase.
        # PicoScope does not use VISA resource, so do not call ScopeBase.__init__().
        self.resource = resource if resource is not None else DummyResource()
        self.timeout = timeout
        self._closed = False
    
        self.scope = psdk.psospa()
        self.scope.open_unit()

        # Default sampling rate. 
        # The actual PicoScope timebase is configured only after set_sampling_rate() is called.
        self._sampling_rate = 1.25e9
        self.timebase = None

        self.samples = 3000
        self.pre_trig_percent = 0
        self._period = (3000, "samples")
        self._delay = (0, "seconds")
        self._capture_samples = 3000
        self._trace_slice = slice(0, 3000)

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
        scale,
        offset,
        threshold_mv=50,
        # Optional parameters for PicoScope configuration
        coupling="DC",
    ):
        if channel not in self.channel_map:
            raise ValueError(f"Channel {channel} is out of range")

        voltage_range, analog_offset = self._channel_voltage_settings(
            scale, offset, voltage_range, psdk.RANGE.mV500)

        pico_ch = self.channel_map[channel]
        self.trigger_channel = pico_ch

        self.scope.set_channel(
            channel=pico_ch,
            coupling=self.coupling_map[coupling],
            range=voltage_range,
            offset=analog_offset,
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
        scale,
        offset,
        period,
        delay=0,
        # Optional parameters for PicoScope configuration
        coupling="AC"
    ):
        """Set the capture window relative to the trigger.

        period accepts seconds or a unit string, including sample counts.
        delay additionally accepts percentages of period; negative values
        select pre-trigger samples. Positive delays are captured and trimmed.
        Sample conversion is deferred until arm(), using the actual rate.
        scale is V/div for eight divisions and must match a supported range
        peak divided by four. offset is the range center voltage in V.
        Specify either scale or voltage_range; the default range is ±20 mV.
        """
        if channel not in self.channel_map:
            raise ValueError(f"Channel {channel} is out of range")

        voltage_range, analog_offset = self._channel_voltage_settings(
            scale, offset, voltage_range, psdk.RANGE.mV20)

        duration = self.decode_time(period)
        start = self.decode_time(delay, allow_percent=True)
        if duration[0] <= 0:
            raise ValueError("period must be positive")
        if start[1] == "percent" and start[0] < -100:
            raise ValueError("delay cannot precede the capture window by more than period")

        pico_ch = self.channel_map[channel]
        self.trace_channel = pico_ch

        self.scope.set_channel(
            channel=pico_ch,
            coupling=self.coupling_map[coupling],
            range=voltage_range,
            offset=analog_offset,
        )

        self._period = duration
        self._delay = start

    def _resolve_capture_window(self):
        # Round to the nearest sample, retaining the original units so rate
        # changes do not change the meaning of the requested window.
        duration, unit = self._period
        count = duration * self._sampling_rate if unit == "seconds" else duration
        delay, unit = self._delay
        if unit == "seconds":
            delay *= self._sampling_rate
        elif unit == "percent":
            delay *= count / 100
        if delay < -count:
            raise ValueError("Negative delay must not exceed period")
        self.samples = round(count)
        if self.samples < 1:
            raise ValueError("period must span at least one sample")
        delay_samples = round(delay)
        trim = max(delay_samples, 0)
        self._capture_samples = self.samples + trim
        self.pre_trig_percent = max(-delay_samples, 0) * 100 / self.samples
        self._trace_slice = slice(trim, trim + self.samples)

    def arm(self):
        if self.timebase is None:
            raise RuntimeError(
                "PicoScope timebase is not configured. "
                "Call set_sampling_rate() before arm()."
                )
        self._resolve_capture_window()
        self.last_trace = None
        self.last_time_axis = None
        self.last_channel_buffer = self.scope.set_data_buffer_for_enabled_channels(
            self._capture_samples
        )

        self.scope.run_block_capture(
            timebase=self.timebase,
            samples=self._capture_samples,
            pre_trig_percent=self.pre_trig_percent,
            segment=0,
        )

        # time.sleep(0.001)

    def capture(self, poll_done=False):
        if self.last_channel_buffer is None:
            raise RuntimeError("PicoScope is not armed. Call arm() before capture().")
        try:
            actual_samples = self.scope.get_values(
                self._capture_samples,
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
                self._capture_samples,
                pre_trig_percent=self.pre_trig_percent,
                ratio=0,
                unit=self.time_unit,
            )[:actual_samples]
            
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
            return np.asarray(self.last_trace)[self._trace_slice].astype(np.int16)

        return np.asarray(self.last_trace)[self._trace_slice]

    def get_last_time_axis(self):
        if self.last_time_axis is None:
            return None
        return self.last_time_axis[self._trace_slice]
    
class DummyResource:
    def close(self):
        pass
