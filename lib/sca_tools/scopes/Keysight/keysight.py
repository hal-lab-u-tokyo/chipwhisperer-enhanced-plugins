
import re

from .MSOX4000 import MSOX4000

# factory method
def KeysightOscilloscope(model, resource, timeout):
    if re.match(r"MSO-X 4104A", model):
        return MSOX4000(model, resource, timeout)
    else:
        raise ValueError(f"Unknown model {model} as Keysight oscilloscope")