# VISA Scope Support for ChipWhisperer API

We have developed a base class to enable VISA-compatible oscilloscopes to integrate seamlessly with the ChipWhisperer API, including routines like `cw.capture_trace`. For detailed usage examples, refer to [Code samples for acquiring traces](../notebooks/acquire_traces_code.ipynb) or the [GUI-based acquisition of traces with a VISA oscilloscope](../notebooks/acquire_traces_with_visa_scope.ipynb).

### Current Limitations
At present, the implementation does not support batch acquisition, meaning it cannot store multiple traces on the oscilloscope memory in a single call to minimize the number of VISA calls. This is a known limitation, and we plan to address it in future updates.

# Connecting to VISA-compatible Oscilloscopes
To connect to a VISA-compatible oscilloscope, you can use the `cw_plugins.scopes.Oscilloscope` method.
This method is a factory function that automatically detects the vendor and model of the connected oscilloscope, determines the appropriate implementation class, and returns an instance of that class.
If the connected oscilloscope is not supported, it raises a `ValueError` with a message indicating that the oscilloscope is not supported.

```Python
from cw_plugins.scopes import Oscilloscope
visaAddress = "USB::...." # replace with the actual VISA address of your oscilloscope
scope = Oscilloscope(visaAddress)
```

To find the VISA address of your oscilloscope, you can use the `pyvisa-shell` command-line tool. 

```bash
$ pyvisa-shell
(visa) list
# shown VISA addresses of connected instruments here
```

# Oscilloscope control API

## `get_num_channels()`
### Description
Returns the number of channels available on the oscilloscope.
### Args
None
### Returns
  - Int: The number of channels available on the oscilloscope.

## `get_sampling_rate()`
### Description
Returns the current sampling rate of the oscilloscope.
### Args
None
### Returns
  - float: The current sampling rate of the oscilloscope.

## `set_sampling_rate(rate)`
### Description
Sets the sampling rate of the oscilloscope to the specified value.
### Args
- `rate` (float or str): Sampling rate in samples per second. It accepts unit-suffixed strings such as `"1.25GS/s"` or `"1250MSPS"` to specify the sampling rate in gigasamples or megasamples per second, respectively.

### Returns
- None
### Raises
- ValueError if the specified rate is out of the supported range for the oscilloscope.
- RuntimeError if the oscilloscope fails to set the sampling rate. For example, if the specified rate is not supported by the oscilloscope, it may return an error.

## `config_trace_channel(channel, scale, offset, period, delay = 0, ...)`
### Description
Configures the specified channel as the trace channel with the given vertical scale, offset, and other optional parameters.
### Args
- `channel` (int): The channel number to configure as the trace channel.
- `scale` (float or str): The vertical scale in V/div; accepts voltage strings such as `"1V"`.
- `offset` (float or str): The voltage at the range center, in volts or a voltage string.
- `period` (float or str): Acquired waveform duration in seconds, a time string such as `"10us"`, or an integer sample count such as `"3000samples"`.
- `delay` (float or str): Capture start relative to the trigger, in seconds, a time string, sample counts, or a percentage of `period`. Negative value means pre-trigger sampling.
It also accepts strings such as `"-50%"` to specify a percentage of the capture period. 
When using a percentage, the value must be between -100% and 100%. For example, `-50%` selects half the trace before the trigger. The default is zero.
- `...`: Additional oscilloscope-specific parameters for configuring the trace channel.
### Returns
None
### Raises
- ValueError if the specified channel is invalid or out of range for the oscilloscope.

## `config_trigger_channel(channel, scale, offset, mode, threshold=None)`
The first three arguments (`channel`, `scale`, and `offset`) match `config_trace_channel()`.

### Description
Configures the specified channel as the trigger channel with the given trigger mode, vertical scale, and offset.
### Args
- `channel` (int): Same as `config_trace_channel()`, the channel number to configure as the trigger channel.
- `scale` (float or str): Same as `config_trace_channel()`, the vertical scale in V/div; accepts voltage strings such as `"2mV"`.
- `offset` (float or str): Same as `config_trace_channel()`, the voltage at the range center, in volts or a voltage string.
- `mode` (str): trigger mode for the trigger channel.
    - `TriggerMode.EDGE_RISE`: Rising edge trigger
    - `TriggerMode.EDGE_FALL`: Falling edge trigger
    - `TriggerMode.EDGE_ANY`: Either rising or falling edge trigger
- `threshold` (float or str, optional): Trigger voltage relative to ground. Accepts volts or a voltage string such as `"500mV"`. The default `None` selects `offset + scale`, one division above the center.

### Returns
None
### Raises
- ValueError if the specified channel is invalid or out of range for the oscilloscope.

## `is_triggered()`
### Description
Checks if the oscilloscope is currently triggered.
### Args
None
### Returns
- bool: True if the oscilloscope is triggered, False otherwise.

## `arm()`
### Description
Starts the oscilloscope to wait for a trigger event.
### Args
None
### Returns
None

## `get_last_trace(as_int)`
### Description
Retrieves the last captured trace from the oscilloscope.
### Args
- `as_int` (bool): If True, the trace is returned as an integer array; otherwise, it is returned as a floating-point array (if possible).
### Returns
- numpy.ndarray: The last captured trace from the oscilloscope, either as an integer array or a floating-point array, depending on the `as_int` argument.

If some error occurs during the VISA communication, it returns `None`.


# Extending Support for Specific Devices
The provided base class is an abstract class, requiring you to create a subclass tailored to your specific VISA-compatible oscilloscope. To open your oscilloscope with `cw_plugins.scopes.Oscilloscope`, additional steps are necessary to enable detection and integration of your device.

## step 1: Implement a subclass to use bender specific VISA commands
The abstract methods are the above oscilloscope control API.
You need to implement these methods using the VISA commands specific to your oscilloscope model.
The implementation examples can be found in the `lib/cw_plugins/scopes` directory.


## step 2: Register your subclass

`lib/cw_plugins.scopes.py` file contains auto discovery mechanism to select the appropriate subclass for the connected oscilloscope.
In that file, `scope_creator` dictionary maps the vendor to corresponding factory function like below:
```python
scope_creator = {
    "AGILENT TECHNOLOGIES": KeysightOscilloscope,
    "RIGOL TECHNOLOGIES": RigolOscilloscope,
}
```

The factory function takes three arguments: `model`, `resource`, and `timeout`. The `model` is a string representing the oscilloscope model, `resource` is the VISA resource string, and `timeout` is the timeout value for VISA operations.
The later two arguments are passed to the constructor of your subclass.
So, you only need to select appropriate subclass based on the `model` argument and return an instance of that subclass.
For Keysight oscilloscopes, it is implemented as follows:
```python
def KeysightOscilloscope(model, resource, timeout):
    if re.match(r"MSO-X 4104A", model):
        return MSOX4000(model, resource, timeout)
    else:
        raise ValueError(f"Unknown model {model} as Keysight oscilloscope")
```
