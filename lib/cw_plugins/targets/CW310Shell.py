###
#   Copyright (C) 2026 University of Tsukuba
#   
#   File:          /lib/cw_plugins/targets/CW310Shell.py
#   Project:       chipwhisperer-enhanced-plugins
#   Author:        Takuya Kojima in University of Tsukuba (tkojima@lila.cs.tsukuba.ac.jp)
#   Created Date:  18-10-2026 17:06:46
#   Last Modified: 09-11-2026 19:38:36
###


from pathlib import Path

from chipwhisperer.capture.targets import CW310

from .CWShellMixin import CWShellMixin
from .utils import vivado_parse_clk_freq

from abc import ABCMeta, abstractmethod

class CW310ShellBase(CWShellMixin,CW310, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.last_key = bytes()
        self.key = bytes()
        self._input_clock = 100E6
        self._bsfile = None

    def _con(self, scope=None, **kwargs):
        """Connect to the target FPGA

            Added arguments:
                hwh_file (str): path to the hardware handoff file (.hwh)
                If not specified, the function tries to find the hwh file in the same directory as the bitstream file.
                In this case, the hwh file name must be the same as the bitstream file name except for the extension.
                Example:
                    bitstream file: /path/to/design.bit
                    hwh file: /path/to/design.hwh
            Please see the base class (CW310) for other arguments.

        """


        if "platform" in kwargs:
            if kwargs["platform"] != "CW310":
                raise ValueError("Currently only the CW310 is supported")

        if "bsfile" in kwargs:
            self._bsfile = kwargs["bsfile"]

        # check if hwh file exists in the same directory as bitstream
        hwh_path = None
        if "hwh_file" not in kwargs:
            if self._bsfile is not None:
                bs_path = Path(self._bsfile)
                hwh_path = bs_path.parent / (bs_path.stem + ".hwh")
        else:
            if "hwh_file" in kwargs and kwargs["hwh_file"] is not None:
                hwh_path = Path(kwargs["hwh_file"])
            kwargs.pop("hwh_file")

        if hwh_path is not None and hwh_path.exists():
            self.memmap = str(hwh_path), "/usb_interface_0"
            self._input_clock = vivado_parse_clk_freq(str(hwh_path), "PLL_CLK1")
        else:
            print("Warning: hwh file not found. It is recommended to specify the hwh file to get the correct memory map and input clock frequency.")


       # we do not use verilog define file.
        # instead, we use the hardware handoff file (.hwh) to get the address map.        
        kwargs["slurp"] = False
        
        super()._con(**kwargs)

        self.init_pll()
        self.reset_interface()
        self.soft_reset()

    def fpga_led_on(self, led):
        """Turn on onbaord LED[0:7]

        Args:
            led (int): LED number to turn on. Must be between 0 and 7
        """
        if led < 0 or led > 7:
            raise ValueError("LED number must be between 0 and 7")
        if self.memmap is None:
            raise RuntimeError("No memory map is available. Load .hwh when connecting to the target")

        try:
            current_leds = self.fpga_read(self.memmap.axi_usr_dip_led.base + 8, 1) # see AMD DS744 datasheet
        except AttributeError:
            raise RuntimeError("AXI GPIO module for LED control is not found in the configured FPGA design. Please check the .hwh file and the design.")

        next_leds = int(current_leds[0]) | (1 << led)
        self.fpga_write(self.memmap.axi_usr_dip_led.base + 8, next_leds)

    def fpga_led_off(self, led):
        """Turn off onbaord LED[0:7]

        Args:
            led (int): LED number to turn off. Must be between 0 and 7
        """
        if led < 0 or led > 7:
            raise ValueError("LED number must be between 0 and 7")
        if self.memmap is None:
            raise RuntimeError("No memory map is available. Load .hwh when connecting to the target")

        try:
            current_leds = self.fpga_read(self.memmap.axi_usr_dip_led.base + 8, 1) # see AMD DS744 datasheet
        except AttributeError:
            raise RuntimeError("AXI GPIO module for LED control is not found in the configured FPGA design. Please check the .hwh file and the design.")

        next_leds = int(current_leds[0]) & ~(1 << led)
        self.fpga_write(self.memmap.axi_usr_dip_led.base + 8, next_leds)


    def init_pll(self):
        """
            Sets up the on-board PLL.
            Especially, PLL_CLK1 is used as a clock source for generating the system clock in the FPGA.
            The actual input clock frequency is obtained from the hwh file if it is specified, otherwise the default value of 100MHz is used.

        """
        self.pll.pll_enable_set(True)
        self.pll.pll_outenable_set(False, 0)
        self.pll.pll_outenable_set(True, 1)
        self.pll.pll_outenable_set(False, 2)

        self.pll.pll_outfreq_set(self._input_clock, 1)


    def reset(self):
        self.reset_interface()
        self.soft_reset()
        self.last_key = bytes()

    def getName(self):
        return "Base Class for CW310 Shell"

    def isDone(self):
        """Check if the target module has finished the encryption"""
        return True

    # Wrapper methods for compatibility with ChipWhisperer.capture_trace
    def set_key(self, key, **kwargs):
        """Set encryption key"""
        self.key = key
        if self.last_key != key:
            self.loadEncryptionKey(key)


    def simpleserial_read(self, cmd, pay_len, **kwargs):
        """Read data from target"""
        if cmd == "r":
            return self.readOutput()
        else:
            raise ValueError("Unknown command {}".format(cmd))

    def simpleserial_write(self, cmd, data, end=None):
        if cmd == 'p':
            self.loadInput(data)
            self.go()
        elif cmd == 'k':
            self.loadEncryptionKey(data)
        else:
            raise ValueError("Unknown command {}".format(cmd))

    def is_done(self):
        return self.isDone()


    # Abstract methods
    @abstractmethod
    def go(self):
        pass

    @abstractmethod
    def getExpected(self):
        """Return expected ciphertext.
            If readed ciphertext is not equal to this value, the encryption is regarded as failed.
        """
        pass

    @abstractmethod
    def loadEncryptionKey(self, key):
        """Load encryption key to the target module"""
        pass

    @abstractmethod
    def loadInput(self, inputtext):
        """Load input text to the target module"""
        pass

    @abstractmethod
    def readOutput(self):
        """Read ciphertext from the target module"""
        pass

