###
#   Copyright (C) 2026 University of Tsukuba
#   
#   File:          /lib/cw_plugins/targets/CW305Shell.py
#   Project:       chipwhisperer-enhanced-plugins
#   Author:        Takuya Kojima in University of Tsukuba (tkojima@lila.cs.tsukuba.ac.jp)
#   Created Date:  18-10-2026 16:36:13
#   Last Modified: 18-10-2026 17:40:46
###


from pathlib import Path
from enum import Enum

from chipwhisperer.capture.targets import CW305

from .CWShellMixin import CWShellMixin

from abc import ABCMeta, abstractmethod


class CW305ShellBase(CWShellMixin,CW305, metaclass=ABCMeta):

    class LED_COLOR(Enum):
        BLUE = 1
        GREEN = 2
        RED = 4

    def __init__(self):
        super().__init__()
        self.last_key = bytes()
        self.key = bytes()

    def _con(self, scope=None, **kwargs):
        """Connect to the target FPGA

            Added arguments:
                hwh_file (str): path to the hardware handoff file (.hwh)
                If not specified, the function tries to find the hwh file in the same directory as the bitstream file.
                In this case, the hwh file name must be the same as the bitstream file name except for the extension.
                Example:
                    bitstream file: /path/to/design.bit
                    hwh file: /path/to/design.hwh
            Please see the base class (CW305) for other arguments.

        """

        if "fpga_id" in kwargs:
            if kwargs["fpga_id"] != "100t":
                raise ValueError("Currently only the 100t is supported")
        if "platform" in kwargs:
            if kwargs["platform"] != "CW305":
                raise ValueError("Currently only the CW305 is supported")

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

        super()._con(**kwargs)


    def setup(self):
        """Default setup for CW305Shell

            The template design assumes that onboard PLL generates 100MHz clock.
            If you change the input clock frequency to the PLL IP (Clocking Wizard IP in Vivado), you need to change the PLL settings.

        """
        self.pll.pll_enable_set(True)
        self.pll.pll_outenable_set(False, 0)
        self.pll.pll_outenable_set(True, 1)
        self.pll.pll_outenable_set(False, 2)
        self.pll.pll_outfreq_set(100E6, 1)
        self.reset()


    def fpga_led_on(self, led):
        """Turn on onbaord LEDs

            Args:
                led (int or CW305ShellBase.LED_COLOR): LED number or color)

                If led is int, it is the LED number (0-2).
                    0: blue (rightmost)
                    1: green (center)
                    2: red (leftmost)
                Anothe way is to use CW305ShellBase.LED_COLOR enum.

            Note:
                This function does not change the state of other LEDs.
                This function needs hardware handoff file (.hwh) to be loaded.
        """

        if self.memmap is None:
            raise RuntimeError("No memory map is not available. Load .hwh when connecting to the target")

        try:
            current_leds = self.fpga_read(self.memmap.axi_gpio.base + 8, 1) # see AMD DS744 datasheet
        except AttributeError:
            raise RuntimeError("AXI GPIO module for LED control is not found in the configured FPGA design. Please check the .hwh file and the design.")

        if isinstance(led, CW305ShellBase.LED_COLOR):
            led = led.value
        if led > 2 or led < 0:
            raise ValueError("LED number must be between 0 and 2")

        next_leds = current_leds | (1 << led)
        self.fpga_write(self.memmap.axi_gpio.base + 8, next_leds)

    def fpga_led_off(self, led):
        """Turn off onbaord LEDs

            The arguments are the same as fpga_led_on.
        """
        if self.memmap is None:
            raise RuntimeError("No memory map is not available. Load .hwh when connecting to the target")

        try:
            current_leds = self.fpga_read(self.memmap.axi_gpio.base + 8, 1)
        except AttributeError:
            raise RuntimeError("AXI GPIO module for LED control is not found in the configured FPGA design. Please check the .hwh file and the design.")
    
        if isinstance(led, CW305ShellBase.LED_COLOR):
            led = led.value
        if led > 2 or led < 0:
            raise ValueError("LED number must be between 0 and 2")

        next_leds = current_leds & ~(1 << led)

        self.fpga_write(self.memmap.axi_gpio.base + 8, next_leds)

    def reset(self):
        self.reset_interface()
        self.soft_reset()
        self.last_key = bytes()

    def getName(self):
        return "Base Class for CW305 Shell"

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

