# VISA Scope Support for ChipWhisperer API

We have developed a base class to enable VISA-compatible oscilloscopes to integrate seamlessly with the ChipWhisperer API, including routines like `cw.capture_trace`. For detailed usage examples, refer to the tutorial notebook [Acquiring Traces](../notebooks/acquire_traces.ipynb).

### Current Limitations
At present, the implementation does not support batch acquisition, meaning it cannot store multiple traces on the oscilloscope memory in a single call to minimize the number of VISA calls. This is a known limitation, and we plan to address it in future updates.

# Extending Support for Specific Devices
The provided base class is an abstract class, requiring you to create a subclass tailored to your specific VISA-compatible oscilloscope. To open your oscilloscope with `cw_plugins.scopes.Oscilloscope`, additional steps are necessary to enable detection and integration of your device.

## step 1: Implement a subclass to use bender specific VISA commands
The abstract methods are listed below.
The implementation examples can be found in the `lib/cw_plugins/scopes` directory.

- `get_num_channels()`: Returns the number of channels available on the oscilloscope.
- `get_sampling_rate()`: Returns the current sampling rate of the oscilloscope.
- `set_sampling_rate(rate)`: Sets the sampling rate of the oscilloscope to the specified value.
- `config_trace_channel(mode, channel, scale, offset, delay = 0)`: Configures the specified channel  as power trace channel with the given vertical scale and offset. As an optional argument, you can specify the delay in seconds to start the trace acquisition after the trigger.
- `config_trigger_channel(mode, channel, scale, offset)`: Configures the specified channel as trigger channel with the given trigger mode, vertical scale and offset.
- `is_triggered()`: Checks if the oscilloscope is currently triggered.
- `arm()`: Starts the oscilloscope to wait for a trigger event.
- `get_last_trace(as_int)`: Retrieves the last captured trace from the oscilloscope. If `as_int` is `True`, the trace is returned as an integer array; otherwise, it is returned as a floating-point array (if possible).

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