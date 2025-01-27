###
#   Copyright (C) 2025 The University of Tokyo
#   
#   File:          /lib/cw_plugins/targets/CW305ShellAESExamples.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  25-01-2025 15:16:34
#   Last Modified: 27-01-2025 15:07:41
###


from .CW305Shell import CW305ShellBase
from Crypto.Cipher import AES
import numpy as np

from pathlib import Path

import warnings

class CW305ShellAES128BitBase(CW305ShellBase):
    def __init__(self):
        super().__init__()
        self.cipher = None
        self.input = bytes()

    def getExpected(self):
        ct = self.cipher.encrypt(bytes(self.input))
        ct = bytearray(ct)
        return ct

    def loadEncryptionKey(self, key):
        self.last_key = key
        self.cipher = AES.new(bytes(key), AES.MODE_ECB)

    def keyLen(self):
        return 16

    def textLen(self):
        return 16

class CW305ShellExampleAES128BitRTL(CW305ShellAES128BitBase):
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

    def __init__(self):
        super().__init__()


    def _con(self, scope=None, implementation="aist", **kwargs):
        name_base = "aes128_aist_rtl" if implementation == "aist" else "aes128_googlevault_rtl"

        use_prebuilt_bitstream = False
        if "bsfile" not in kwargs:
            kwargs["bsfile"] = Path(__file__).parent / "bitstreams" / "cw305" / name_base + ".bit"
            use_prebuilt_bitstream = True

        if "hwh_file" not in kwargs and use_prebuilt_bitstream:
            kwargs["hwh_file"] = Path(__file__).parent / "hwh_files" / "cw305" / name_base + ".hwh"

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
        else:
            raise ValueError(f"Unknown implementation {implementation}")



    def getName(self):
        return "CW305 Shell AES-128 RTL Example"

    def run_aist_core(self):
        self.fpga_write(self.address_base + self.AIST_CORE.ADDRESS_MAP["control"], \
                         [self.AIST_CORE.KEY_READY_BIT])
        self.fpga_write(self.address_base + self.AIST_CORE.ADDRESS_MAP["control"], \
                            [self.AIST_CORE.PT_READY_BIT])

    def run_google_core(self):
        # self.fpga_write(self.address_base + self.GOOGLE_CORE.ADDRESS_MAP["control"], \
        #                  [self.GOOGLE_CORE.ENC_START_BIT ])
        # use external trigger
        self.usb_trigger_toggle()

    # implement abstract methods
    def go(self):
        self.run_impl()

    def loadEncryptionKey(self, key):
        # call parent method to update cipher object
        super().loadEncryptionKey(key)
        key_words = [int.from_bytes(key[4*i:4*i+4], byteorder='big') for i in range(4)]
        self.fpga_write(self.key_address, key_words)

    def loadInput(self, inputtext):
        self.input = inputtext
        input_words = [int.from_bytes(inputtext[4*i:4*i+4], byteorder='big') for i in range(4)]

        self.fpga_write(self.pt_address, input_words)

    def readOutput(self):
        ct_words = self.fpga_read(self.ct_address, 4)
        ct = b''
        for w in ct_words:
            ct += int(w).to_bytes(4, byteorder='big')

        return np.frombuffer(ct, dtype=np.uint8)

class CW305ShellExampleAES128BitHLS(CW305ShellAES128BitBase):
    CTRL_ADDRESS_MAP = {
        "control": 0x0,
        "key_offset": 0x10,
        "plaintext_offset": 0x18,
        "ciphertext_offset": 0x20,
    }
    CTRL_AP_START = 0x1
    CTRL_AP_DONE = 0x2
    CTRL_AP_IDLE = 0x4
    CTRL_AP_READY = 0x8
    CTRL_AP_CONTINUE = 0x10

    def __init__(self):
        super().__init__()

    def setup(self):
        super().setup()
        self.fpga_write(self.core_address + self.CTRL_ADDRESS_MAP["key_offset"], [self.key_address])
        self.fpga_write(self.core_address + self.CTRL_ADDRESS_MAP["plaintext_offset"], [self.plaintext_address])
        self.fpga_write(self.core_address + self.CTRL_ADDRESS_MAP["ciphertext_offset"], [self.ciphertext_address])
        self.fpga_write(self.core_address + self.CTRL_ADDRESS_MAP["control"], [self.CTRL_AP_CONTINUE])
        print("setup done")

    def _con(self, scope=None, **kwargs):

        use_prebuilt_bitstream = False
        if "bsfile" not in kwargs:
            kwargs["bsfile"] = Path(__file__).parent / "bitstreams" / "cw305" / "aes128_hls.bit"
            use_prebuilt_bitstream = True

        if "hwh_file" not in kwargs and use_prebuilt_bitstream:
            kwargs["hwh_file"] = Path(__file__).parent / "hwh_files" / "cw305" / "aes128_hls.hwh"


        super()._con(scope, **kwargs)
        try:
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
        return "CW305 Shell AES-128 HLS Example"

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
            self.fpga_write(self.core_address + self.CTRL_ADDRESS_MAP["control"], [self.CTRL_AP_START])
        else:
            raise RuntimeError("HLS IP is not idle")

    def loadEncryptionKey(self, key):
        # call parent method to update cipher object
        super().loadEncryptionKey(key)
        key_words = [int.from_bytes(key[4*i:4*i+4], byteorder='little') for i in range(4)]

        self.fpga_write(self.key_address, key_words)

    def loadInput(self, inputtext):
        self.input = inputtext
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
