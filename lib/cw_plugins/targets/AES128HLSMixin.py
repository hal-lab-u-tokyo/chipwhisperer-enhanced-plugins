###
#   Copyright (C) 2026 University of Tsukuba
#   
#   File:          /lib/cw_plugins/targets/AES128HLSMixin.py
#   Project:       chipwhisperer-enhanced-plugins
#   Author:        Takuya Kojima in University of Tsukuba (tkojima@lila.cs.tsukuba.ac.jp)
#   Created Date:  21-10-2026 00:34:06
#   Last Modified: 21-10-2026 00:40:56
###

from .AES128Mixin import AES128Mixin
import random
import numpy as np
import warnings
from pathlib import Path
from abc import ABCMeta, abstractmethod

class AES128HLSMixin(AES128Mixin,metaclass=ABCMeta):
    CTRL_ADDRESS_MAP = {
        "control": 0x0,
        "key_offset": 0x10,
        "plaintext_offset": 0x18,
        "ciphertext_offset": 0x20,
        "rotate": 0x28
    }
    CTRL_AP_START = 0x1
    CTRL_AP_DONE = 0x2
    CTRL_AP_IDLE = 0x4
    CTRL_AP_READY = 0x8
    CTRL_AP_CONTINUE = 0x10

    def __init__(self):
        super().__init__()
        self.rsm_impl = False

    def setup(self):
        super().setup()
        self.fpga_write(self.core_address + self.CTRL_ADDRESS_MAP["key_offset"], [self.key_address])
        self.fpga_write(self.core_address + self.CTRL_ADDRESS_MAP["plaintext_offset"], [self.plaintext_address])
        self.fpga_write(self.core_address + self.CTRL_ADDRESS_MAP["ciphertext_offset"], [self.ciphertext_address])
        self.fpga_write(self.core_address + self.CTRL_ADDRESS_MAP["control"], [self.CTRL_AP_CONTINUE])

    @staticmethod
    @abstractmethod
    def getPlatformName():
        pass

    def _con(self, scope=None, implementation = "naive", **kwargs):

        if implementation == "naive":
            name_base = "aes128_hls"
        elif implementation == "rsm":
            name_base = "aes128_rsm_hls"
            self.rsm_impl = True
        else:
            raise ValueError(f"Unknown implementation {implementation}")

        use_prebuilt_bitstream = False
        if "bsfile" not in kwargs:
            kwargs["bsfile"] = (Path(__file__).parent / "bitstreams" / self.getPlatformName() / f"{name_base}").with_suffix(".bit")
            use_prebuilt_bitstream = True

        if "hwh_file" not in kwargs and use_prebuilt_bitstream:
            kwargs["hwh_file"] = (Path(__file__).parent / "hwh_files" / self.getPlatformName() / f"{name_base}").with_suffix(".hwh")


        super()._con(scope, **kwargs)
        try:
            if implementation == "rsm":
                self.core_address = self.memmap.RSM_AES128Encrypt_0.base
            elif implementation == "naive":
                self.core_address = self.memmap.AES128Encrypt_0.base

            self.bram_address = self.memmap.axi_bram_ctrl_1.base
        except AttributeError:
            warnings.warn("Error loading hardware handoff file. Using default address map.")
            self.core_address = 0x8000_0000
            self.bram_address = 0xC000_0000

        self.key_address = self.bram_address
        self.plaintext_address = self.bram_address + 0x10
        self.ciphertext_address = self.bram_address + 0x20

        self.setup()


    def getName(self):
        return f"{self.getPlatformName()} Shell AES-128 HLS Example"

    def get_status(self):
        stat = self.fpga_read(self.core_address + self.CTRL_ADDRESS_MAP["control"], 1)[0]
        ap_start = (stat & self.CTRL_AP_START) != 0
        ap_done = (stat & self.CTRL_AP_DONE) != 0
        ap_idle = (stat & self.CTRL_AP_IDLE) != 0
        ap_ready = (stat & self.CTRL_AP_READY) != 0
        ap_continue = (stat & self.CTRL_AP_CONTINUE) != 0

        return {"ap_start": ap_start, "ap_done": ap_done, "ap_idle": ap_idle, "ap_ready": ap_ready, "ap_continue": ap_continue}

    # implement abstract methods
    def go(self):
        stat = self.get_status()
        if stat["ap_idle"]:
            if self.rsm_impl:
                rotate = random.randint(0, 15)
                self.fpga_write(self.core_address + self.CTRL_ADDRESS_MAP["rotate"], rotate)
            self.fpga_write(self.core_address + self.CTRL_ADDRESS_MAP["control"], [self.CTRL_AP_START])
        else:
            raise RuntimeError("HLS IP is not idle")

    def loadEncryptionKey(self, key):
        self.registerEncryptionKey(key)
        key_words = [int.from_bytes(key[4*i:4*i+4], byteorder='little') for i in range(4)]

        self.fpga_write(self.key_address, key_words)

    def loadInput(self, inputtext):
        self.registerInput(inputtext)
        input_words = [int.from_bytes(inputtext[4*i:4*i+4], byteorder='little') for i in range(4)]

        self.fpga_write(self.plaintext_address, input_words)

    def readOutput(self):
        stat = self.get_status()
        if not stat["ap_done"]:
            raise RuntimeError("HLS IP is not done")

        ct_words = self.fpga_read(self.ciphertext_address, 4)
        ct = b''
        for w in ct_words:
            ct += int(w).to_bytes(4, byteorder='little')
        self.fpga_write(self.core_address + self.CTRL_ADDRESS_MAP["control"], [self.CTRL_AP_CONTINUE])
        return np.frombuffer(ct, dtype=np.uint8)


    def isDone(self):
        stat = self.get_status()
        return stat["ap_done"]
