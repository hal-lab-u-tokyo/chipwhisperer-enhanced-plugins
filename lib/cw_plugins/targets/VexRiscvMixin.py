###
#   Copyright (C) 2026 University of Tsukuba
#   
#   File:          /lib/cw_plugins/targets/VexRiscvMixin.py
#   Project:       chipwhisperer-enhanced-plugins
#   Author:        Takuya Kojima in University of Tsukuba (tkojima@lila.cs.tsukuba.ac.jp)
#   Created Date:  22-10-2026 17:07:52
#   Last Modified: 22-10-2026 18:43:36
###

from abc import ABCMeta, abstractmethod

from .utils import vivado_parse_memmap, ParseError, MemoryMap

import atexit
import warnings
from elftools.elf.elffile import ELFFile
from collections import namedtuple
import numpy as np
from pathlib import Path


class VexRiscvMixin(metaclass=ABCMeta):
    Segment = namedtuple("Segment", ["offset", "size", "binary_size", "data"])

    # Peripheral address
    RECV_BUF_DATA_ADDR = 0x0
    RECV_BUF_STAT_ADDR = 0x4
    SEND_BUF_DATA_ADDR = 0x8
    SEND_BUF_STAT_ADDR = 0xC

    CHUNK_SIZE = 16

    @staticmethod
    @abstractmethod
    def getPlatformName():
        pass
    
    @staticmethod
    @abstractmethod
    def isBsfileSupported():
        pass

    def __init__(self):
        self._core_stop_on_disconnect = False
        super().__init__()
        self.segments = []
        self.control_address = 0x4000_0000
        self.memmap_core = MemoryMap()
        self.memmap_core.add_range("axi_buffer_0", 0xA200_0000, 0xA200_FFFF)


    def _con(self, scope = None, program = None, **kwargs):
        if program is None:
            raise ValueError("program argument must be specified")

        if "verbose" in kwargs:
            verbose = kwargs["verbose"]
            kwargs.pop("verbose")
        else:
            verbose = False

        self.debug_print = lambda x: print("[INFO]", x) if verbose else lambda x: None

        use_prebuilt_bitstream = False
        name_base = "VexRiscv"
        if self.isBsfileSupported():
            use_prebuilt_bitstream = "bsfile" not in kwargs
            if use_prebuilt_bitstream:
                kwargs["bsfile"] = (Path(__file__).parent / "bitstreams" / self.getPlatformName() / f"{name_base}").with_suffix(".bit")
        else:
            if "bsfile" in kwargs:
                raise ValueError("bsfile is not supported for this platform")

        if "hwh_file" not in kwargs and use_prebuilt_bitstream:
            kwargs["hwh_file"] = (Path(__file__).parent / "hwh_files" / self.getPlatformName() / f"{name_base}").with_suffix(".hwh")

        # parent class _con() will parse the hwh file and set self.memmap, which is expected to be the memory map of the interface module (e.g., axi_buffer_0)
        super()._con(scope, **kwargs)

        if "hwh_file" in kwargs:
            hwh_file = kwargs["hwh_file"]
            try:
                memmap_core = vivado_parse_memmap(hwh_file, "/VexRiscv_Core_0")
                self.memmap_core = memmap_core
                self.control_address = self.memmap.VexRiscv_Core_0.base
            except (AttributeError, ParseError, FileNotFoundError) as E:
                warnings.warn("Error loading hardware handoff file: " + E.args[0] + ". Using default address map." )

        
        self.segments = []
        self.loadProgram(program)


        # core reset
        self._core_stop_on_disconnect = True
        atexit.register(self.close)
        self.core_start()
        self.core_stop()

        self.boot()
        self.flush()

        self.core_start()


    def loadProgram(self, program):
        self.segments = []
        with open(program, 'rb') as f:
            self.elf = ELFFile(f)
            for i in range(self.elf.num_segments()):
                seg = self.elf.get_segment(i)
                if seg.header.p_type == 'PT_LOAD':
                    if seg.header.p_memsz == 0:
                        continue
                else:
                    continue
                f.seek(seg.header.p_offset)
                data = f.read(seg.header.p_filesz)
                if len(data) % 4 != 0:
                    data += b"\x00" * (4 - len(data) % 4)
                self.segments.append(self.Segment(seg.header.p_vaddr, seg.header.p_memsz, seg.header.p_filesz, data))
        self.debug_print(f"Loaded {len(self.segments)} segments")

    def boot(self):
        self.debug_print("boot start")
        for seg in self.segments:
            base = seg.offset
            word_data = np.frombuffer(seg.data, dtype="<u4")
            for i in range(0, len(word_data), self.CHUNK_SIZE):
                chunk = [int(d) for d in word_data[i:i+self.CHUNK_SIZE]]
                self.fpga_write(base + i * 4, chunk)
        self.debug_print("boot end")

    # core control
    def core_stop(self):
        self.debug_print("Core stop")
        self.fpga_write(self.control_address, [0x0])

    def core_start(self):
        self.debug_print("Core start")
        self.fpga_write(self.control_address, [0x1])


    # external intrrupt signal
    def assert_interrupt(self):
        self.fpga_write(self.control_address + 4, [0x1])

    def deassert_interrupt(self):
        self.fpga_write(self.control_address + 4, [0x0])

    def reset(self):
        super().reset()
        self.core_stop()

    def dis(self):
        atexit.unregister(self.close)
        stop_core = self._core_stop_on_disconnect
        self._core_stop_on_disconnect = False
        try:
            if self.getPlatformName() in ("cw305", "cw310"):
                naeusb = getattr(self, "_naeusb", None)
                backend = getattr(naeusb, "usbtx", None)
                stop_core = stop_core and getattr(backend, "handle", None) is not None
            if stop_core:
                self.core_stop()
        finally:
            try:
                super().dis()
            finally:
                self.connectStatus = False

    def close(self):
        if self._core_stop_on_disconnect:
            self.dis()

    def flush(self):
        super().flush()
        addr = self.memmap_core.axi_buffer_0.base + \
                self.RECV_BUF_DATA_ADDR
        while not self.is_recv_buffer_empty():
            _ = self.fpga_read(addr, 1)[0]
        addr = self.memmap_core.axi_buffer_0.base + \
                self.SEND_BUF_DATA_ADDR
        while not self.is_send_buffer_empty():
            _ = self.fpga_read(addr, 1)[0]

    # buffer control
    def is_send_buffer_full(self):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.SEND_BUF_STAT_ADDR
        return bool(self.fpga_read(addr, 1)[0] & 0x100)

    def is_send_buffer_empty(self):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.SEND_BUF_STAT_ADDR
        return self.fpga_read(addr, 1)[0] & 0xFF == 1

    def is_recv_buffer_full(self):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.RECV_BUF_STAT_ADDR
        return bool(self.fpga_read(addr, 1)[0] & 0x100)

    def is_recv_buffer_empty(self):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.RECV_BUF_STAT_ADDR
        return self.fpga_read(addr, 1)[0] & 0xFF == 1

    def get_send_buffer_bytes(self):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.SEND_BUF_STAT_ADDR
        stat = self.fpga_read(addr, 1)[0]
        return (stat & 0xFFFF0000) >> 16

    def get_recv_buffer_bytes(self):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.RECV_BUF_STAT_ADDR
        stat = self.fpga_read(addr, 1)[0]
        return (stat & 0xFFFF0000) >> 16

    def send_bytes(self, data):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.SEND_BUF_DATA_ADDR
        for b in data:
            # wait until send buffer is not full
            while self.is_send_buffer_full():
                pass
            self.fpga_write(addr, [b])

    def recv_bytes(self, length):
        data = bytearray()
        addr = self.memmap_core.axi_buffer_0.base + \
                self.RECV_BUF_DATA_ADDR
        for i in range(length):
            # wait until recv buffer is not empty
            while self.is_recv_buffer_empty():
                pass
            data.append(self.fpga_read(addr, 1)[0])
        return bytes(data)

