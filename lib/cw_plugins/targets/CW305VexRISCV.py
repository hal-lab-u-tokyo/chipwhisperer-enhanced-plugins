
from .CW305Shell import CW305ShellBase
from .utils import vivado_parse_memmap, ParseError, MemoryMap

import warnings
from elftools.elf.elffile import ELFFile
from collections import namedtuple
import numpy as np
from abc import ABCMeta
from pathlib import Path

class CW305VexRISCVBase(CW305ShellBase, metaclass=ABCMeta):
    Segment = namedtuple("Segment", ["offset", "size", "binary_size", "data"])

    # Peripheral address
    RECV_BUF_DATA_ADDR = 0x0
    RECV_BUF_STAT_ADDR = 0x4
    SEND_BUF_DATA_ADDR = 0x8
    SEND_BUF_STAT_ADDR = 0xC

    CHUNK_SIZE = 16

    def __init__(self):
        super().__init__()
        # use default address map
        self.periph_address_base = 0xA200_0000
        self.periph_address_offset = {
            "RECV_BUF_DATA_ADDR": 0x0000,
            "RECV_BUF_STAT_ADDR": 0x0004,
            "SEND_BUF_DATA_ADDR": 0x0008,
            "SEND_BUF_STAT_ADDR": 0x000C,
        }

    def _con(self, scope = None, program = None, **kwargs):

        if "verbose" in kwargs:
            verbose = kwargs["verbose"]
            kwargs.pop("verbose")
        else:
            verbose = False

        self.debug_print = lambda x: print("[INFO]", x) if verbose else lambda x: None

        self.segments = []
        self.loadProgram(program)

        if program is None:
            raise ValueError("program argument must be specified")

        super()._con(scope, **kwargs)

        hwh_file = kwargs.get("hwh_file", None)

        try:
            self.control_address = self.memmap.VexRiscv_Core_0.base
        except AttributeError:
            warnings.warn("Error loading hardware handoff file. Using default address map.")
            self.control_address = 0x4000_0000

        if not hwh_file is None:
            try:
                self.memmap_core = vivado_parse_memmap(hwh_file, "/VexRiscv_Core_0")
            except (AttributeError, ParseError, FileNotFoundError) as E:
                warnings.warn("Error loading hardware handoff file: " + E.args[0] + ". Using default address map." )
                self.memmap_core = MemoryMap()
                self.memmap_core.add_range("axi_buffer_0", 0xA200_0000, 0xA200_FFFF)

        # system reset
        self.setup()

        # core reset
        self.core_start()
        self.core_stop()

        self.boot()
        self.flush()

        self.debug_print("Core start")
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
        self.fpga_write(self.control_address, 0x0)

    def core_start(self):
        self.fpga_write(self.control_address, 0x1)


    # external intrrupt signal
    def assert_interrupt(self):
        self.fpga_write(self.control_address + 4, 0x1)

    def deassert_interrupt(self):
        self.fpga_write(self.control_address + 4, 0x0)

    def reset(self):
        super().reset()
        self.core_stop()

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
        return self.fpga_read(addr, 1)[0] & 0xFF00 == 1

    def is_send_buffer_empty(self):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.SEND_BUF_STAT_ADDR
        return self.fpga_read(addr, 1)[0] & 0xFF == 1

    def is_recv_buffer_full(self):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.RECV_BUF_STAT_ADDR
        return self.fpga_read(addr, 1)[0] & 0xFF00 == 1

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

    def go(self):
        raise NotImplementedError

    def getExpected(self):
        raise NotImplementedError

    def loadEncryptionKey(self, key):
        raise NotImplementedError

    def loadInput(self, inputtext):
        raise NotImplementedError

    def readOutput(self):
        raise NotImplementedError


