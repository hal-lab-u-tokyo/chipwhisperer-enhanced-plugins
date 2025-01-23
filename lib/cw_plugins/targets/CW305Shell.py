###
#   Copyright (C) 2025 The University of Tokyo
#   
#   File:          /lib/cw_plugins/targets/CW305Shell.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  22-01-2025 08:34:28
#   Last Modified: 24-01-2025 03:21:07
###

from pathlib import Path
import asyncio
from enum import Enum

from chipwhisperer.capture.targets import CW305
import numpy as np

from .utils import vivado_parse_memmap


class CW305Shell(CW305):
    # flit format constants
    FLIT_READ_ADDR_UPPER=0
    FLIT_READ_ADDR_LOWER=1
    FLIT_WRITE_ADDR_UPPER=2
    FLIT_WRITE_ADDR_LOWER=3
    FLIT_WRITE_DATA_UPPER=4
    FLIT_WRITE_DATA_LOWER=5
    FLIT_RESET=6

    RESPONSE_CMD_OK = 0
    RESPONSE_READY = 3
    RESPONSE_NOT_READY = 0xFF

    class LED_COLOR(Enum):
        BLUE = 1
        GREEN = 2
        RED = 4

    def __init__(self):
        super(CW305Shell, self).__init__()
        self.bytecount_size = 0 # avoid address increment in base class implementation
        self.memmap = None
        self.timeout = 1

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
            self.memmap = vivado_parse_memmap(str(hwh_path), "/usb_interface_0")

        super(CW305Shell, self)._con(**kwargs)

    def reset_interface(self):
        """Force reset the stuck state of the interface module"""
        _ = super().fpga_read(0xFFFF, 1)

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
        self.reset_interface()
        self.soft_reset()

    def __make_flit(self, type, payload, data_len = 0):
        """Helper function to make flit"""
        # payload is 16-bit
        if payload > 0xffff:
            raise ValueError("payload is too large")
        flit = payload
        flit |= (data_len << 16)
        flit |= (type << 20)
        # calculate even parity
        parity = 0
        for i in range(24):
            parity ^= (flit >> i) & 1
        flit |= (parity << 24)
        return flit

    async def __wait_until_not_ready(self):
        """Wait until the FPGA responds some value other than NOT_READY"""
        try:
            while True:
                resp = super().fpga_read(0x0, 1)[0]
                if resp != CW305Shell.RESPONSE_NOT_READY:
                    return resp
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            return -1
        except asyncio.TimeoutError:
            return -1


    def fpga_write(self, addr, data):
        """Write data to FPGA based on the shell communication protocol

            Args:
                addr (int): address to write data
                data (int or list of int): data to write
                In the case of a list, the data length must be less than or equal to 16.

            Note:
                All data is written as 32-bit data.
                If multiple data are written, the address is incremented by 4 for each data.

            Raises:
                ValueError: if data length is greater than 16
                RuntimeError: if the shell interface does not respond correctly

        """
        # check if data is iterable
        if not hasattr(data, "__iter__"):
            data = [data]
        if len(data) > 16:
            raise ValueError("data is too large")

        addr_upper = addr >> 16
        addr_lower = addr & 0xffff
        flits = []
        flits.append(self.__make_flit(CW305Shell.FLIT_WRITE_ADDR_UPPER, addr_upper, len(data)-1))
        flits.append(self.__make_flit(CW305Shell.FLIT_WRITE_ADDR_LOWER, addr_lower))
        for d in data:
            data_upper = d >> 16
            if data_upper > 0xffff:
                Warning("sending data exceeding 32-bit limit. It will be truncated")
            data_lower = d & 0xffff
            flits.append(self.__make_flit(CW305Shell.FLIT_WRITE_DATA_UPPER, data_upper))
            flits.append(self.__make_flit(CW305Shell.FLIT_WRITE_DATA_LOWER, data_lower))
        for flit in flits:
            addr = flit >> 8
            binary = flit & 0xff
            super().fpga_write(addr, [binary])

        resp = asyncio.run(asyncio.wait_for(self.__wait_until_not_ready(), timeout=self.timeout))
        if resp != CW305Shell.RESPONSE_CMD_OK:
            if resp < 0:
                msg = f"FPGA write timed out"
            else:
                msg = f"FPGA write got unexpected response 0x{resp:02x}"
            self.reset_interface()
            raise RuntimeError(msg)

    def fpga_read(self, addr, data_len):
        """Read data from FPGA based on the shell communication protocol

            Args:
                addr (int): address to read data
                data_len (int): number of 32-bit data to read
                The data length must be less than or equal to 16.

            Returns:
                numpy array (data type = numpy.uint32)

            Raises:
                ValueError: if data length is greater than 16
                RuntimeError: if the shell interface does not respond correctly

        """

        if data_len > 16 or data_len < 0:
            raise ValueError("data_len must be between 1 and 16")
        addr_upper = addr >> 16
        addr_lower = addr & 0xffff
        flits = []
        flits.append(self.__make_flit(CW305Shell.FLIT_READ_ADDR_UPPER, addr_upper, data_len-1))
        flits.append(self.__make_flit(CW305Shell.FLIT_READ_ADDR_LOWER, addr_lower))
        for flit in flits:
            addr = flit >> 8
            data = flit & 0xff
            super().fpga_write(addr, [data])
        resp = asyncio.run(asyncio.wait_for(self.__wait_until_not_ready(), timeout=self.timeout))
        if resp != CW305Shell.RESPONSE_READY:
            if resp < 0:
                msg = f"FPGA read timed out"
            else:
                msg = f"FPGA read got unexpected response 0x{resp:02x}"
            self.reset_interface()
            raise RuntimeError(msg)

        # get data
        binary_data = super().fpga_read(0x0, data_len*4)
        # decode as little endian
        data = np.frombuffer(binary_data, "<u4")
        resp = super().fpga_read(0x0, 1)[0]
        if resp != CW305Shell.RESPONSE_CMD_OK:
            self.reset_interface()
            raise RuntimeError(f"unexpected response 0x{resp:02x}", data)

        return data

    def soft_reset(self):
        """Soft reset the FPGA

            This function sends a reset command to the FPGA.
            Then, the interface module resets the other modules in the FPGA.
            To reset the stuck state of the interface module, call reset_interface() instead.

        """
        reset_cmd_flit = self.__make_flit(CW305Shell.FLIT_RESET, 0)
        addr = reset_cmd_flit >> 8
        data = reset_cmd_flit & 0xff
        super().fpga_write(addr, [data])
        resp = asyncio.run(asyncio.wait_for(self.__wait_until_not_ready(), timeout=self.timeout))
        if resp != CW305Shell.RESPONSE_CMD_OK:
            if resp < 0:
                msg = f"Soft reset timed out"
            else:
                msg = f"Soft reset got unexpected response 0x{resp:02x}"
            self.reset_interface()
            raise RuntimeError(msg)

    def fpga_led_on(self, led):
        """Turn on onbaord LEDs

            Args:
                led (int or CW305Shell.LED_COLOR): LED number or color)

                If led is int, it is the LED number (0-2).
                    0: blue (rightmost)
                    1: green (center)
                    2: red (leftmost)
                Anothe way is to use CW305Shell.LED_COLOR enum.

            Note:
                This function does not change the state of other LEDs.
                This function needs hardware handoff file (.hwh) to be loaded.
        """

        if self.memmap is None:
            raise RuntimeError("No memory map is not available. Load .hwh when connecting to the target")

        current_leds = self.fpga_read(self.memmap.axi_gpio.base + 8, 1) # see AMD DS744 datasheet

        if isinstance(led, CW305Shell.LED_COLOR):
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

        current_leds = self.fpga_read(self.memmap.axi_gpio.base + 8, 1)
        if isinstance(led, CW305Shell.LED_COLOR):
            led = led.value
        if led > 2 or led < 0:
            raise ValueError("LED number must be between 0 and 2")

        next_leds = current_leds & ~(1 << led)

        self.fpga_write(self.memmap.axi_gpio.base + 8, next_leds)
