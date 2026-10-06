###
#   Copyright (C) 2024 The University of Tokyo
#   
#   File:          /lib/cw_plugins/targets/SakuraXVexRISCV.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  13-07-2024 16:20:31
#   Last Modified: 15-11-2025 15:23:18
###

from .SakuraXShell import SakuraXShellControlBase
from .utils import vivado_parse_memmap, ParseError, MemoryMap

import warnings
from elftools.elf.elffile import ELFFile
from collections import namedtuple
import numpy as np
from abc import ABCMeta
from pathlib import Path

class SakuraXVexRISCVControlBase(SakuraXShellControlBase, metaclass=ABCMeta):
    Segment = namedtuple("Segment", ["offset", "size", "binary_size", "data"])

    # Peripheral address
    RECV_BUF_DATA_ADDR = 0x0
    RECV_BUF_STAT_ADDR = 0x4
    SEND_BUF_DATA_ADDR = 0x8
    SEND_BUF_STAT_ADDR = 0xC

    # def get_periph_address(self, name):
    #     return self.periph_address_base + self.periph_address_offset[name]

    CHUNK_SIZE = 16

    def __init__(self, ser, program, hwh_file = None, verbose = False):
        super().__init__(ser)
        self.debug_print = lambda x: print("[INFO]", x) if verbose else lambda x: None
        # system reset
        self.reset_command()
        self.segments = []
        self.loadProgram(program)

        # use default address map
        self.control_address = 0x4000_0000
        self.memmap_core = MemoryMap()
        self.memmap_core.add_range("axi_buffer_0", 0xA200_0000, 0xA200_FFFF)

        if not hwh_file is None:
            try:
                memmap_ctrl = vivado_parse_memmap(hwh_file, "/controller_AXI_0")
                memmap_core = vivado_parse_memmap(hwh_file, "/VexRiscv_Core_0")
                self.control_address = memmap_ctrl.VexRiscv_Core_0.base
                self.memmap_core = memmap_core
            except (AttributeError, ParseError, FileNotFoundError) as E:
                warnings.warn("Error loading hardware handoff file: " + E.args[0] + ". Using default address map." )

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
                self.write_data(base + i * 4, chunk)
        self.debug_print("boot end")

    # core control
    def core_stop(self):
        self.write_data(self.control_address, [0x0])

    def core_start(self):
        self.write_data(self.control_address, [0x1])


    # external intrrupt signal
    def assert_interrupt(self):
        self.write_data(self.control_address + 4, [0x1])

    def deassert_interrupt(self):
        self.write_data(self.control_address + 4, [0x0])

    def reset(self):
        super().reset()
        self.core_stop()

    def close(self):
        self.core_stop()
        super().close()

    def flush(self):
        super().flush()
        addr = self.memmap_core.axi_buffer_0.base + \
                self.RECV_BUF_DATA_ADDR
        while not self.is_recv_buffer_empty():
            _ = self.read_data(addr, 1)[0]
        addr = self.memmap_core.axi_buffer_0.base + \
                self.SEND_BUF_DATA_ADDR
        while not self.is_send_buffer_empty():
            _ = self.read_data(addr, 1)[0]

    # buffer control
    def is_send_buffer_full(self):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.SEND_BUF_STAT_ADDR
        return self.read_data(addr, 1)[0] & 0xFF00 == 1

    def is_send_buffer_empty(self):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.SEND_BUF_STAT_ADDR
        return self.read_data(addr, 1)[0] & 0xFF == 1

    def is_recv_buffer_full(self):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.RECV_BUF_STAT_ADDR
        return self.read_data(addr, 1)[0] & 0xFF00 == 1

    def is_recv_buffer_empty(self):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.RECV_BUF_STAT_ADDR
        return self.read_data(addr, 1)[0] & 0xFF == 1

    def get_send_buffer_bytes(self):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.SEND_BUF_STAT_ADDR
        stat = self.read_data(addr, 1)[0]
        return (stat & 0xFFFF0000) >> 16

    def get_recv_buffer_bytes(self):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.RECV_BUF_STAT_ADDR
        stat = self.read_data(addr, 1)[0]
        return (stat & 0xFFFF0000) >> 16

    def send_bytes(self, data):
        addr = self.memmap_core.axi_buffer_0.base + \
                self.SEND_BUF_DATA_ADDR
        for b in data:
            # wait until send buffer is not full
            while self.is_send_buffer_full():
                pass
            self.write_data(addr, [b])

    def recv_bytes(self, length):
        data = bytearray()
        addr = self.memmap_core.axi_buffer_0.base + \
                self.RECV_BUF_DATA_ADDR
        for i in range(length):
            # wait until recv buffer is not empty
            while self.is_recv_buffer_empty():
                pass
            data.append(self.read_data(addr, 1)[0])
        return bytes(data)

    def send_key(self, key : bytes):
        raise NotImplementedError

    def send_plaintext(self, plaintext : bytes):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def read_ciphertext(self, byte_len : int = 8):
        raise NotImplementedError

