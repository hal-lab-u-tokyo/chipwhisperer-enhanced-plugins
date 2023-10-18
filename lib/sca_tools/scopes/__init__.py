from .base import TriggerMode
from .scopes import Oscilloscope

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