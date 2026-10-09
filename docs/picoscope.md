# PicoScope 3000E

The `PicoScope3000E` class depends on `pypicosdk` and has been tested with the PicoScope 3418E.
Install the native PicoSDK driver for your device and operating system as well as this package's Python dependencies. 
Please follow the official installation instructions [page](https://www.picotech.com/library/knowledge-bases/oscilloscopes/pypicosdk-get-started).

The wrapper uses the `psospa` backend; other models and driver backends have not been verified.
Connect directly through the SDK using `PicoScope3000E()`.
The VISA `Oscilloscope(visaAddr)` factory does not open PicoScope devices.

## API Overview
Most of the API is shared with [the VISA oscilloscope wrapper](visa_scope.md), but some methods have been modified to accommodate the PicoScope's unique features. The following methods are available:
- `set_sampling_rate(rate)`: Sets the sampling rate of the oscilloscope to the specified value. 
However, the actual sampling rate is set when the trace channel is configured, so this method only stores the requested rate.
If you specify unacceptable values, the PicoSDK may return an error when configuring the trace channel.

- `config_trigger_channel(mode, channel, scale, offset, threshold=None, coupling="DC", probe_scale=10.0)`: Configures the specified channel as the trigger channel with the given trigger mode, vertical scale, offset, and other optional parameters.
The PicoScope SDK only accepts voltage range instead of V/div.
However, considering 10 vertical divisions of PicoScope software, the wrapper converts V/div to voltage range by multiplying by 5.
PicoScope cannot detect the probe attenuation factor, so the wrapper by default assumes a 10:1 probe. The `probe_scale` argument allows you to specify the actual probe attenuation factor (1:1 or 10:1) for the trigger channel.
- `config_trace_channel(channel, scale, offset, period, delay=0, coupling="AC", resolution=10)`: Configures the specified channel as the trace channel with the given vertical scale, offset, and other optional parameters.
The former argumens are the same as trigger channel configuration.
Unlike conventional oscilloscopes, the PicoScope SDK accepts limited range of offset values.
Therefore, AC coupling is generally recommended to avoid over-range measurements.
3000E series oscilloscopes support 8- or 10-bit acquisition, which can be selected with the `resolution` argument. The default is 10 bits.
The configured resulution affects the maximum sampling rate.
That is why the wrapper requires `set_sampling_rate()` to be called before `config_trace_channel()`.

## Example usage


```python
import chipwhisperer as cw
from cw_plugins.scopes import TriggerMode
from cw_plugins.scopes.PicoScope import PicoScope3000E

scope = PicoScope3000E()
try:
    # Store the requested rate before configuring the trace channel.
    scope.set_sampling_rate("1.25GS/s")
    scope.config_trigger_channel(
        TriggerMode.EDGE_RISE, "B", scale="1V", offset=0,
        threshold="1V", 
        # optional parameters peculiar to the PicoScope.
        coupling="DC", probe_scale=10.0,
    )
    scope.config_trace_channel(
        "A", scale="4mV", offset=0,
        period="3000samples", delay="-50%", coupling="AC", resolution=10,
    )

    # target is an already connected as a ChipWhisperer target.
    # trace = cw.capture_trace(scope, target, plaintext, key)

    ktp = cw.ktp.Basic()
    key = ktp.next_key()
    traceCount = 10
    project = cw.create_project("your project.cwp")
    for _ in range(traceCount):
        pt = ktp.next_text()
        target.loadInput(pt)
        trace = cw.capture_trace(scope, target, pt, key)
        if trace is None:
            print("Trace capture failed.")
            break
        else:
            project.traces.append(trace)

finally:
    scope.close()

```

