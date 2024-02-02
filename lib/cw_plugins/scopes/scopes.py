import pyvisa

from .Keysight import KeysightOscilloscope
from .Rigol import RigolOscilloscope
from .base import ScopeBase

# factory method for each vendor
scope_creator = {
    "AGILENT TECHNOLOGIES": KeysightOscilloscope,
    "RIGOL TECHNOLOGIES": RigolOscilloscope,
}

def Oscilloscope(visaAddr, timeout=10) -> ScopeBase:
    """Create an oscilloscope object from the given VISA address.
        visaAddr: VISA address of the oscilloscope
        timeout: timeout in s
    """
    rm = pyvisa.ResourceManager()
    inst = rm.open_resource(visaAddr)
    try:
        inst_info = inst.query("*IDN?")
    except pyvisa.errors.VisaIOError:
        raise RuntimeError(f"Failed to connect to {visaAddr}")
    vendor, model, serialNum, revcode = inst_info.split(",")

    if vendor not in scope_creator:
        raise ValueError(f"Unknown vendor {vendor} as oscilloscope")
    else:
        return scope_creator[vendor](model, inst, timeout)

