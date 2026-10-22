###
#   Copyright (C) 2026 University of Tsukuba
#   
#   File:          /lib/cw_plugins/targets/CWShellMixin.py
#   Project:       chipwhisperer-enhanced-plugins
#   Author:        Takuya Kojima in University of Tsukuba (tkojima@lila.cs.tsukuba.ac.jp)
#   Created Date:  18-10-2026 16:37:05
#   Last Modified: 18-10-2026 17:38:14
###

import time
import numpy as np
from os import PathLike
from .utils import vivado_parse_memmap,MemoryMap

class CWShellMixin:
    """Common mixin function for CW305Shell and CW310Shell.
        This class mainly defines the communication protocol between the host and the shell part on the FPGA.
    """

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

    def __init__(self):
        # CW305 or CW310 base class initialization is called in the derived class
        super().__init__()
        self.bytecount_size = 0 # avoid address increment in base class implementation
        self.timeout = 1
        self._memmap = None


    @property
    def memmap(self):
        return self._memmap

    @memmap.setter
    def memmap(self, value: MemoryMap | tuple[str | PathLike, str]):
        if isinstance(value, MemoryMap):
            self._memmap = value
        elif isinstance(value, tuple) and len(value) == 2:
            hwh_file, module_name = value
            self._memmap = vivado_parse_memmap(hwh_file, module_name)
        else:
            raise ValueError("memmap must be a MemoryMap instance or a tuple of (hwh_file, module_name)")

    def __raw_fpga_read(self, addr, leadlen):
        return super().fpga_read(addr, leadlen)

    def __raw_fpga_write(self, addr, data):
        super().fpga_write(addr, data)

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

    def __wait_until_not_ready(self):
        """Wait until the FPGA responds some value other than NOT_READY"""
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            resp = self.__raw_fpga_read(0x0, 1)[0]
            if resp != self.RESPONSE_NOT_READY:
                return resp
            time.sleep(0.01)

        return -1 # timeout

    def reset_interface(self):
        """Force reset the stuck state of the interface module"""
        _ = self.__raw_fpga_read(0xFFFF, 1)

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
        flits.append(self.__make_flit(self.FLIT_WRITE_ADDR_UPPER, addr_upper, len(data)-1))
        flits.append(self.__make_flit(self.FLIT_WRITE_ADDR_LOWER, addr_lower))
        for d in data:
            data_upper = d >> 16
            if data_upper > 0xffff:
                Warning("sending data exceeding 32-bit limit. It will be truncated")
            data_lower = d & 0xffff
            flits.append(self.__make_flit(self.FLIT_WRITE_DATA_UPPER, data_upper))
            flits.append(self.__make_flit(self.FLIT_WRITE_DATA_LOWER, data_lower))
        for flit in flits:
            addr = flit >> 8
            binary = flit & 0xff
            self.__raw_fpga_write(addr, [binary])

        resp = self.__wait_until_not_ready()
        if resp != self.RESPONSE_CMD_OK:
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
        flits.append(self.__make_flit(self.FLIT_READ_ADDR_UPPER, addr_upper, data_len-1))
        flits.append(self.__make_flit(self.FLIT_READ_ADDR_LOWER, addr_lower))
        for flit in flits:
            addr = flit >> 8
            data = flit & 0xff
            self.__raw_fpga_write(addr, [data])
        resp = self.__wait_until_not_ready()
        if resp != self.RESPONSE_READY:
            if resp < 0:
                msg = f"FPGA read timed out"
            else:
                msg = f"FPGA read got unexpected response 0x{resp:02x}"
            self.reset_interface()
            raise RuntimeError(msg)

        # get data
        binary_data = self.__raw_fpga_read(0x0, data_len*4)
        # decode as little endian
        data = np.frombuffer(binary_data, "<u4")
        resp = self.__raw_fpga_read(0x0, 1)[0]
        if resp != self.RESPONSE_CMD_OK:
            self.reset_interface()
            raise RuntimeError(f"unexpected response 0x{resp:02x}", data)

        return data

    def soft_reset(self):
        """Soft reset the FPGA

            This function sends a reset command to the FPGA.
            Then, the interface module resets the other modules in the FPGA.
            To reset the stuck state of the interface module, call reset_interface() instead.

        """
        reset_cmd_flit = self.__make_flit(self.FLIT_RESET, 0)
        addr = reset_cmd_flit >> 8
        data = reset_cmd_flit & 0xff
        self.__raw_fpga_write(addr, [data])
        resp = self.__wait_until_not_ready()
        if resp != self.RESPONSE_CMD_OK:
            if resp < 0:
                msg = f"Soft reset timed out"
            else:
                msg = f"Soft reset got unexpected response 0x{resp:02x}"
            self.reset_interface()
            raise RuntimeError(msg)