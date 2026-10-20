###
#   Copyright (C) 2026 University of Tsukuba
#   
#   File:          /lib/cw_plugins/targets/AES128RTLMixin.py
#   Project:       chipwhisperer-enhanced-plugins
#   Author:        Takuya Kojima in University of Tsukuba (tkojima@lila.cs.tsukuba.ac.jp)
#   Created Date:  21-10-2026 00:33:44
#   Last Modified: 21-10-2026 00:33:46
###

from .AES128Mixin import AES128Mixin
import random
import numpy as np
import warnings
from pathlib import Path
from abc import ABCMeta, abstractmethod

class AES128RTLMixin(AES128Mixin,metaclass=ABCMeta):
    class AIST_CORE():
        ADDRESS_MAP = {
            "key": 0x0,
            "plaintext": 0x10,
            "ciphertext": 0x20,
            "control": 0x30,
        }
        KEY_READY_BIT = 0x2
        PT_READY_BIT = 0x1
    class GOOGLE_CORE():
        ADDRESS_MAP = {
            "key": 0x0,
            "plaintext": 0x20,
            "ciphertext": 0x30,
            "control": 0x40,
        }
        AES128_SIZE = 0
        ENC_START_BIT = 0x1

    class RSM_CORE():
        ADDRESS_MAP = {
            "key": 0x0,
            "plaintext": 0x10,
            "ciphertext": 0x20,
            "control": 0x30,
            "rotate": 0x34
        }
        KEY_READY_BIT = 0x2
        PT_READY_BIT = 0x1

    NAME_BASE_DICT = {
        "aist": "aes128_aist_rtl",
        "google": "aes128_googlevault_rtl",
        "rsm": "aes128_rsm_rtl"
    }

    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def getPlatformName():
        pass

    def _con(self, scope=None, implementation="aist", **kwargs):
        name_base = self.NAME_BASE_DICT[implementation]

        use_prebuilt_bitstream = False
        if "bsfile" not in kwargs:
            kwargs["bsfile"] = (Path(__file__).parent / "bitstreams" / self.getPlatformName() / name_base).with_suffix(".bit")
            use_prebuilt_bitstream = True

        if "hwh_file" not in kwargs and use_prebuilt_bitstream:
            kwargs["hwh_file"] = (Path(__file__).parent / "hwh_files" / self.getPlatformName() / name_base).with_suffix(".hwh")

        super()._con(scope, **kwargs)
        try:
            self.address_base = self.memmap.aes_rtl_core_0.base
        except AttributeError:
            warnings.warn("Error loading hardware handoff file. Using default address map.")
            self.address_base = 0x8000_0000

        self.setup()

        if implementation == "aist":
            self.key_address = self.address_base + self.AIST_CORE.ADDRESS_MAP["key"]
            self.pt_address = self.address_base + self.AIST_CORE.ADDRESS_MAP["plaintext"]
            self.ct_address = self.address_base + self.AIST_CORE.ADDRESS_MAP["ciphertext"]
            self.run_impl = self.run_aist_core
        elif implementation == "google":
            self.key_address = self.address_base + self.GOOGLE_CORE.ADDRESS_MAP["key"]
            self.pt_address = self.address_base + self.GOOGLE_CORE.ADDRESS_MAP["plaintext"]
            self.ct_address = self.address_base + self.GOOGLE_CORE.ADDRESS_MAP["ciphertext"]
            self.fpga_write(self.address_base + self.GOOGLE_CORE.ADDRESS_MAP["control"], \
                         [self.GOOGLE_CORE.AES128_SIZE << 1 ])
            self.run_impl = self.run_google_core
        elif implementation == "rsm":
            self.key_address = self.address_base + self.RSM_CORE.ADDRESS_MAP["key"]
            self.pt_address = self.address_base + self.RSM_CORE.ADDRESS_MAP["plaintext"]
            self.ct_address = self.address_base + self.RSM_CORE.ADDRESS_MAP["ciphertext"]
            self.run_impl = self.run_rsm_core
        else:
            raise ValueError(f"Unknown implementation {implementation}")


    def getName(self):
        return f"{self.getPlatformName()} Shell AES-128 RTL Example"

    def run_aist_core(self):
        self.fpga_write(self.address_base + self.AIST_CORE.ADDRESS_MAP["control"], \
                         [self.AIST_CORE.KEY_READY_BIT])
        self.fpga_write(self.address_base + self.AIST_CORE.ADDRESS_MAP["control"], \
                            [self.AIST_CORE.PT_READY_BIT])

    def run_google_core(self):
        # use external trigger
        self.usb_trigger_toggle()

    def run_rsm_core(self):
        rotate = random.randint(0, 15)
        self.fpga_write(self.address_base + self.RSM_CORE.ADDRESS_MAP["rotate"], [rotate])
        self.fpga_write(self.address_base + self.RSM_CORE.ADDRESS_MAP["control"], \
                         [self.RSM_CORE.KEY_READY_BIT])
        self.fpga_write(self.address_base + self.RSM_CORE.ADDRESS_MAP["control"], \
                            [self.RSM_CORE.PT_READY_BIT])

    # implement abstract methods
    def go(self):
        self.run_impl()

    def loadEncryptionKey(self, key):
        self.registerEncryptionKey(key)
        key_words = [int.from_bytes(key[4*i:4*i+4], byteorder='big') for i in range(4)]
        self.fpga_write(self.key_address, key_words)

    def loadInput(self, inputtext):
        self.registerInput(inputtext)
        input_words = [int.from_bytes(inputtext[4*i:4*i+4], byteorder='big') for i in range(4)]

        self.fpga_write(self.pt_address, input_words)

    def readOutput(self):
        ct_words = self.fpga_read(self.ct_address, 4)
        ct = b''
        for w in ct_words:
            ct += int(w).to_bytes(4, byteorder='big')

        return np.frombuffer(ct, dtype=np.uint8)