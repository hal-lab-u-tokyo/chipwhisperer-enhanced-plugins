###
#   Copyright (C) 2026 University of Tsukuba
#   
#   File:          /lib/cw_plugins/targets/CW310Shell.py
#   Project:       chipwhisperer-enhanced-plugins
#   Author:        Takuya Kojima in University of Tsukuba (tkojima@lila.cs.tsukuba.ac.jp)
#   Created Date:  18-10-2026 17:06:46
#   Last Modified: 20-10-2026 20:15:18
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
        if "hwh_file" not in kwargs:
            bs_path = Path(self._bsfile)
            hwh_path = bs_path.parent / (bs_path.stem + ".hwh")
        else:
            hwh_path = Path(kwargs["hwh_file"])
            kwargs.pop("hwh_file")

        if hwh_path.exists():
            self.memmap = str(hwh_path), "/usb_interface_0"
            self._input_clock = vivado_parse_clk_freq(str(hwh_path), "PLL_CLK1")

        super()._con(**kwargs)


    def setup(self):
        """
            Default setup for CW310Shell. Mostly, this function sets up the on-board PLL.
            Especially, PLL_CLK1 is used as a clock source for generating the system clock in the FPGA.
            The actual input clock frequency is obtained from the hwh file if it is specified, otherwise the default value of 100MHz is used.

        """
        self.pll.pll_enable_set(True)
        self.pll.pll_outenable_set(False, 0)
        self.pll.pll_outenable_set(True, 1)
        self.pll.pll_outenable_set(False, 2)

        self.pll.pll_outfreq_set(self._input_clock, 1)

        self.reset()

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

