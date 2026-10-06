# PicoScope 3000E

The `PicoScope3000E` wrapper uses `pypicosdk` and has been tested with the PicoScope 3418E.
Install the native PicoSDK driver for your device and operating system as well as this package's Python dependencies. 
Please follow the official installation instructions [page](https://www.picotech.com/library/knowledge-bases/oscilloscopes/pypicosdk-get-started).

The wrapper uses the `psospa` backend; other models and driver backends have not been verified.
Connect directly through the SDK using `PicoScope3000E()`.
The VISA `Oscilloscope(visaAddr)` factory does not open PicoScope devices.

```python
import chipwhisperer as cw
import pypicosdk as psdk
from cw_plugins.scopes import PicoScope3000E, TriggerMode

scope = PicoScope3000E()
try:
    # Configure channels before selecting the sampling rate.
    scope.config_trigger_channel(
        TriggerMode.EDGE_RISE, "B", threshold_mv=50,
        coupling="DC", voltage_range=psdk.RANGE.mV500,
    )
    scope.config_trace_channel(
        "A", coupling="AC", voltage_range=psdk.RANGE.mV20,
        samples=3000, pre_trig_percent=50,
    )
    actual_rate = scope.set_sampling_rate(1.25e9)

    # target is an already connected ChipWhisperer target.
    # trace = cw.capture_trace(scope, target, plaintext, key)

    scope.arm()
    # Trigger the target here before retrieving the capture.
    if scope.capture():
        raise RuntimeError("PicoScope capture failed")
    waveform = scope.get_last_trace()
    time_axis = scope.get_last_time_axis()
finally:
    scope.close()
```


Channels use the strings `"A"`, `"B"`, `"C"`, and `"D"`.
Select the input range with `voltage_range`, coupling with `"AC"` or `"DC"`, and the trigger threshold in millivolts with `threshold_mv`.
The shared API's `scale`, `offset`, `delay`, and `impedance` arguments are currently unused by this wrapper.

Sampling rates are in samples per second. `set_sampling_rate()` returns the actual hardware rate, which can differ from the requested rate, and must be called before `arm()`.
Trace length can be specified with `samples`, or with `period` in seconds after setting the sampling rate. `pre_trig_percent` selects the percentage of samples before the trigger (0–100).

Waveforms default to millivolts and the time axis defaults to nanoseconds.
Set `scope.output_unit` to `"mv"`, `"v"`, or `"adc"` before acquisition to select millivolts, volts, or raw ADC counts.
`get_last_trace(as_int=True)` casts the selected output to `int16`; select `"adc"` to obtain raw integer counts.

`capture()` returns `False` on success and `True` on retrieval failure, matching the ChipWhisperer convention.
Capture completion is handled by the SDK's `get_values()`; `is_triggered()` is a placeholder and `poll_done` is unused.
The constructor's `resource` and `timeout` arguments are retained for API compatibility and do not configure VISA communication or an SDK capture timeout.
