# This code was made and tested based on 3418E.

from cw_plugins.scopes.base import ScopeBase, TriggerMode
import math
import ctypes
from numbers import Real
import numpy as np
import pypicosdk as psdk
import time

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

    # Ten vertical divisions: the bipolar range peak spans five divisions.
    scale_map = {
        0.010 / 5: psdk.RANGE.mV10,
        0.020 / 5: psdk.RANGE.mV20,
        0.050 / 5: psdk.RANGE.mV50,
        0.100 / 5: psdk.RANGE.mV100,
        0.200 / 5: psdk.RANGE.mV200,
        0.500 / 5: psdk.RANGE.mV500,
        1.0 / 5: psdk.RANGE.V1,
        2.0 / 5: psdk.RANGE.V2,
        5.0 / 5: psdk.RANGE.V5,
        10.0 / 5: psdk.RANGE.V10,
        20.0 / 5: psdk.RANGE.V20,
    }
    SAMPLING_RATE_REL_TOL = 0.01

    def _channel_voltage_settings(self, scale, offset):
        scale = self.decode_voltage(scale)
        offset = self.decode_voltage(offset)
        for supported_scale, supported_range in self.scale_map.items():
            if math.isclose(scale, supported_scale, rel_tol=1e-12, abs_tol=0):
                voltage_range = supported_range
                break
        else:
            raise ValueError(
                f"Unsupported scale {scale!r}; supported V/div values: {list(self.scale_map)}"
            )
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
        self.resolution = None

        # Default sampling rate. 
        # The actual timebase is resolved when the trace channel is configured.
        self._sampling_rate = 1.25e9
        self.requested_sampling_rate = None
        self.timebase = None

        self.samples = 3000
        self.pre_trig_percent = 0
        self._period = (3000, "samples")
        self._delay = (0, "seconds")
        self._capture_samples = 3000
        self._trace_slice = slice(0, 3000)

        self.trace_channel = psdk.CHANNEL.A
        self.trigger_channel = psdk.CHANNEL.B

        self.output_unit = "v"
        self.time_unit = "ns"

        self.last_channel_buffer = None
        self.last_time_axis = None
        self.last_trace = None
        self.last_raw_trace = None
        self._adc_buffers = None
        

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
        """Store the requested rate for the next config_trace_channel()."""
        rate = self.decode_sampling_rate(rate)
        self.requested_sampling_rate = rate
        self.timebase = None

    def _configure_timebase(self):
        """Resolve the timebase after the capture channels are enabled."""
        rate = self.requested_sampling_rate
        self.timebase = None
        if rate is None:
            raise RuntimeError("Call set_sampling_rate() before config_trace_channel().")

        timebase = self.scope.sample_rate_to_timebase(
            sample_rate=rate / 1e9,
            unit=psdk.SAMPLE_RATE.GSPS,
        )

        actual_rate = self.scope.get_actual_sample_rate()
        if (not math.isfinite(actual_rate) or actual_rate <= 0
                or not math.isclose(actual_rate, rate,
                                    rel_tol=self.SAMPLING_RATE_REL_TOL, abs_tol=0)):
            raise RuntimeError(
                f"PicoScope sampling rate differs from requested rate by more than "
                f"{self.SAMPLING_RATE_REL_TOL:.0%}: requested {rate:g} S/s, "
                f"actual {actual_rate:g} S/s"
            )
        self.timebase = timebase
        self._sampling_rate = actual_rate

        return self._sampling_rate

    
    def config_trigger_channel(
        self,
        mode,
        channel,
        scale,
        offset,
        threshold=None,
        # Optional parameters for PicoScope configuration
        coupling="DC",
        probe_scale=10.0,
    ):
        """Set a GND-referenced threshold in V, accepting voltage strings.

        None selects one division above the range center (offset + scale).
        probe_scale is the probe attenuation (1 for 1:1, 10 for 10:1 (default)).
        scale selects the input range as V/div over ten divisions, independently
        of probe_scale. offset is the input range center voltage; threshold
        refers to the probe tip and is scaled by the SDK.
        """
        if channel not in self.channel_map:
            raise ValueError(f"Channel {channel} is out of range")

        if (isinstance(probe_scale, bool) or not isinstance(probe_scale, Real)
                or not math.isfinite(probe_scale) or probe_scale < 1):
            raise ValueError("probe_scale must be a finite attenuation factor >= 1")

        scale = self.decode_voltage(scale)
        offset = self.decode_voltage(offset)
        threshold = self.decode_voltage(offset + scale if threshold is None else threshold)
        voltage_range, analog_offset = self._channel_voltage_settings(
            scale, offset)
        direction = self.slope_map[mode]

        pico_ch = self.channel_map[channel]
        self.trigger_channel = pico_ch
        self._adc_buffers = None

        self.scope.set_channel(
            channel=pico_ch,
            coupling=self.coupling_map[coupling],
            range=voltage_range,
            offset=analog_offset,
            probe_scale=probe_scale,
        )

        self.scope.set_simple_trigger(
            channel=pico_ch,
            threshold=threshold * 1000,
            direction=direction,
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
        coupling="AC",
        resolution=10,
    ):
        """Set the capture window relative to the trigger.

        period accepts seconds or a unit string, including sample counts.
        delay additionally accepts percentages of period; negative values
        select pre-trigger samples. Positive delays are captured and trimmed.
        Sample conversion is deferred until arm(), using the actual rate.
        scale is V/div for ten divisions and must match a supported range
        peak divided by five. offset is the range center voltage in V.
        scale and offset also accept voltage strings.
        resolution selects 8- or 10-bit acquisition (default 10) for the device.
        Voltage output uses mV for ranges below +/-1 V, otherwise V;
        output_unit records the selected unit. Raw ADC output is unaffected.
        """
        if channel not in self.channel_map:
            raise ValueError(f"Channel {channel} is out of range")

        if isinstance(resolution, bool) or resolution not in (8, 10):
            raise ValueError("resolution must be 8 or 10 bits")

        voltage_range, analog_offset = self._channel_voltage_settings(scale, offset)

        duration = self.decode_time(period)
        start = self.decode_time(delay, allow_percent=True)
        if duration[0] <= 0:
            raise ValueError("period must be positive")
        if start[1] == "percent" and start[0] < -100:
            raise ValueError("delay cannot precede the capture window by more than period")

        self.timebase = None
        self._adc_buffers = None
        self.last_channel_buffer = None
        if resolution != self.resolution:
            sdk_resolution = psdk.RESOLUTION._8BIT if resolution == 8 else psdk.RESOLUTION._10BIT
            self.scope.set_device_resolution(sdk_resolution)
            self.resolution = resolution

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
        self._configure_timebase()
        self.output_unit = "mv" if self.decode_voltage(scale) * 5 < 1 else "v"
        self._adc_buffers = None
        if coupling == "AC":
            # Channel settings reach the hardware at RunBlock. Apply them
            # once here, before real captures, and let AC coupling settle.
            try:
                self._run_block_capture(1, 0)
                time.sleep(1)
            except Exception:
                self.timebase = None
                raise
            finally:
                self.scope.stop()

    def _run_block_capture(self, samples, pre_samples):
        # psospaRunBlock expects uint64 sample counts and a double* for
        # timeIndisposedMs. pyPicoSDK 1.7.5 passes an int32* for the latter.
        # We do not need that estimate, so pass the documented NULL pointer.
        self.scope._call_attr_function(
            "RunBlock", self.scope.handle,
            ctypes.c_uint64(pre_samples), ctypes.c_uint64(samples - pre_samples),
            ctypes.c_uint32(self.timebase), None, ctypes.c_uint64(0), None, None,
        )

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
                "Call set_sampling_rate() then config_trace_channel() before arm()."
                )
        self._resolve_capture_window()
        self.last_trace = None
        self.last_raw_trace = None
        self.last_time_axis = None
        if self._adc_buffers is None:
            self._adc_buffers = self.scope.set_data_buffer_for_enabled_channels(
                self._capture_samples
            )
        self.last_channel_buffer = self._adc_buffers
        pre_samples = round(self._capture_samples * self.pre_trig_percent / 100)
        self._run_block_capture(self._capture_samples, pre_samples)

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

            # Keep driver-owned buffers registered across captures; return
            # snapshots so the next acquisition cannot overwrite old traces.
            self.last_channel_buffer = {
                ch: buffer[:actual_samples].copy()
                for ch, buffer in self._adc_buffers.items()
            }

            raw_trace = self.last_channel_buffer[self.trace_channel].copy()

            if self.output_unit == "mv":
                self.last_channel_buffer = self.scope.adc_to_mv(
                    self.last_channel_buffer
                )
            elif self.output_unit == "v":
                self.last_channel_buffer = self.scope.adc_to_volts(
                    self.last_channel_buffer
                )
            else:
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
            self.last_raw_trace = raw_trace
            return False

        except Exception as e:
            print(
                f"PicoScope capture failed: {type(e).__name__}: {e} "
                f"(samples={self.samples}, timebase={self.timebase}, "
                f"trace_channel={self.trace_channel}, output_unit={self.output_unit})"
            )
            return True

    def get_last_trace(self, as_int=False):
        """Return raw ADC counts if as_int, otherwise voltage.

        Both representations preserve the same capture window. output_unit
        records v or mv, selected from the configured trace range, and does
        not affect raw counts.
        """
        if self.last_trace is None:
            return None

        if as_int:
            return self.last_raw_trace[self._trace_slice]

        return np.asarray(self.last_trace)[self._trace_slice]

    def get_last_time_axis(self):
        if self.last_time_axis is None:
            return None
        return self.last_time_axis[self._trace_slice]
    
class DummyResource:
    def close(self):
        pass
