
import re

from .MSO8000 import MSO8000
from .DHO900 import DHO900

def RigolOscilloscope(model, resource, timeout):
    if re.match(r'^MSO(8064|8104|8204)$', model):
        return MSO8000(model, resource, timeout)
    elif re.match(r"^DHO(924)S", model):
        return DHO900(model, resource, timeout)
    else:
        raise ValueError(f"Unknown model {model} as Keysight oscilloscope")
