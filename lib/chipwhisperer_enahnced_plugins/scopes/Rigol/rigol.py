
import re

from .MSO8000 import MSO8000

def RigolOscilloscope(model, resource, timeout):
    if re.match(r'^MSO(8064|8104|8204)$', model):
        return MSO8000(model, resource, timeout)
    else:
        raise ValueError(f"Unknown model {model} as Keysight oscilloscope")
