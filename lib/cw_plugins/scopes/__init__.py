from .base import TriggerMode
from .scopes import Oscilloscope


def __getattr__(name):
    # Loading the Pico SDK is only necessary when a PicoScope is requested.
    if name == "PicoScope3000E":
        from .PicoScope import PicoScope3000E
        return PicoScope3000E
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# unit translation utilities
def milliVolt(value):
    return value * 1e-3

def milliSecond(value):
    return value * 1e-3

def microSecond(value):
    return value * 1e-6

def nanoSecond(value):
    return value * 1e-9

def giga(value):
    return value * 1e9
