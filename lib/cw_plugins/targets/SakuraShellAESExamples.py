###
#   Copyright (C) 2024 The University of Tokyo
#   
#   File:          /lib/cw_plugins/targets/SakuraShellAESExamples.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)#   Created Date:  13-07-2024 15:38:26
#   Last Modified: 25-01-2025 15:21:48
###

from .SakuraXShell import SakuraXShellBase, SakuraXShellControlBase
from Crypto.Cipher import AES
from .utils import vivado_parse_memmap, ParseError
import warnings
from pathlib import Path

from abc import ABCMeta

class SakuraXShellAES(SakuraXShellBase, metaclass=ABCMeta):

    def __init__(self, ):
        super().__init__()
        self.cipher = None
        self.input = bytes()

    def textLen(self):
        return 16

    def keyLen(self):
        return 16

    def getExpected(self):
        ct = self.cipher.encrypt(bytes(self.input))
        ct = bytearray(ct)
        return ct

    def loadEncryptionKey(self, key):
        self.ctrl.send_key(bytes(key))
        self.last_key = key
        self.cipher = AES.new(bytes(key), AES.MODE_ECB)

    def loadInput(self, inputtext):
        self.input = inputtext
        self.ctrl.send_plaintext(bytes(inputtext))



# Example Implementations for Sakura-X Shell

class SakuraXShellAES128BitRTLControl(SakuraXShellControlBase):
    # for AIST RTL AES Core
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
    def __init__(self, ser, hwh_file = None, implementation = "aist", **kwargs):
        super().__init__(ser)
        self.reset_command()
        self.address_base = None


        if hwh_file is None:
            name_base = "aes128_aist_rtl" if implementation == "aist" else "aes128_googlevault_rtl"
            hwh_file = Path(__file__).parent / "hwh_files" / "sakura-x" / name_base + ".hwh"
        try:
            memmap = vivado_parse_memmap(hwh_file, "/controller_AXI_0")
            self.address_base = memmap.aes_rtl_core_0.base
        except (AttributeError, ParseError, FileNotFoundError) as E:
            warnings.warn("Error loading hardware handoff file: " + E.args[0] + ". Using default address map." )

        if self.address_base is None:
            self.address_base = 0x8000_0000

        if implementation == "aist":
            self.key_address = self.address_base + self.AIST_CORE.ADDRESS_MAP["key"]
            self.pt_address = self.address_base + self.AIST_CORE.ADDRESS_MAP["plaintext"]
            self.ct_address = self.address_base + self.AIST_CORE.ADDRESS_MAP["ciphertext"]
            self.run_impl = self.run_aist_core
        elif implementation == "google":
            self.key_address = self.address_base + self.GOOGLE_CORE.ADDRESS_MAP["key"]
            self.pt_address = self.address_base + self.GOOGLE_CORE.ADDRESS_MAP["plaintext"]
            self.ct_address = self.address_base + self.GOOGLE_CORE.ADDRESS_MAP["ciphertext"]
            self.write_data(self.address_base + self.GOOGLE_CORE.ADDRESS_MAP["control"], \
                         [self.GOOGLE_CORE.AES128_SIZE << 1 ])
            self.run_impl = self.run_google_core
        else:
            raise ValueError(f"Unknown implementation {implementation}")




    def send_key(self, key : bytes):
        key_words = [int.from_bytes(key[4*i:4*i+4], byteorder='big') for i in range(4)]
        self.write_data(self.key_address, key_words)

    def send_plaintext(self, plaintext : bytes):
        pt_words = [int.from_bytes(plaintext[4*i:4*i+4], byteorder='big') for i in range(4)]
        self.write_data(self.pt_address, pt_words)

    def run_aist_core(self):
        self.write_data(self.address_base + self.AIST_CORE.ADDRESS_MAP["control"], \
                         [self.AIST_CORE.KEY_READY_BIT])
        self.write_data(self.address_base + self.AIST_CORE.ADDRESS_MAP["control"], \
                            [self.AIST_CORE.PT_READY_BIT])

    def run_google_core(self):
        self.write_data(self.address_base + self.GOOGLE_CORE.ADDRESS_MAP["control"], \
                         [self.GOOGLE_CORE.ENC_START_BIT ])

    def run(self):
        self.run_impl()

    def read_ciphertext(self, byte_len : int = 8):
        read_words = self.read_data(self.ct_address, byte_len // 4)
        ct = b''
        for w in read_words:
            ct += w.to_bytes(4, byteorder='big')
        return ct


class SakuraXShellExampleAES128BitRTL(SakuraXShellAES):
    def getName(self):
        return "Sakura-X Shell AES128 Bit RTL Example"

    def getControl(self, **kwargs) -> SakuraXShellControlBase:
        return SakuraXShellAES128BitRTLControl(self.ser, **kwargs)

class SakuraXShellAES128BitHLSControl(SakuraXShellControlBase):
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

    def __init__(self, ser, hwh_file, **kwargs):
        self.control_address = None
        bram_address = None
        if hwh_file is None:
            hwh_file = Path(__file__).parent / "hwh_files" / "sakura-x" / "aes128_hls.hwh"
        try:
            memmap = vivado_parse_memmap(hwh_file, "/controller_AXI_0")
            self.control_address = memmap.AES128Encrypt_0.base
            bram_address = memmap.axi_bram_ctrl_1.base
        except (AttributeError, ParseError, FileNotFoundError) as E:
            warnings.warn("Error loading hardware handoff file: " + E.args[0] + ". Using default address map." )

        if self.control_address is None:
            self.control_address = 0x8000_0000
        if bram_address is None:
            bram_address = 0xC000_0000

        self.key_address = bram_address
        self.plaintext_address = bram_address + 0x10
        self.ciphertext_address = bram_address + 0x20

        super().__init__(ser)
        self.reset_command()

        self.write_data(self.control_address + self.CTRL_ADDRESS_MAP["key_offset"], [self.key_address])
        self.write_data(self.control_address + self.CTRL_ADDRESS_MAP["plaintext_offset"], [self.plaintext_address])
        self.write_data(self.control_address + self.CTRL_ADDRESS_MAP["ciphertext_offset"], [self.ciphertext_address])

        self.write_data(self.control_address + self.CTRL_ADDRESS_MAP["control"], [self.CTRL_AP_CONTINUE])


    def get_status(self):
        stat = self.read_data(self.control_address + self.CTRL_ADDRESS_MAP["control"], 1)[0]
        ap_start = (stat & self.CTRL_AP_START) != 0
        ap_done = (stat & self.CTRL_AP_DONE) != 0
        ap_idle = (stat & self.CTRL_AP_IDLE) != 0
        ap_ready = (stat & self.CTRL_AP_READY) != 0
        ap_continue = (stat & self.CTRL_AP_CONTINUE) != 0
        return {"ap_start": ap_start, "ap_done": ap_done, "ap_idle": ap_idle, "ap_ready": ap_ready, "ap_continue": ap_continue}


    def send_key(self, key : bytes):
        data = []
        for i in range(4):
            data.append(int.from_bytes(key[4*i:4*i+4], byteorder='little'))
        self.write_data(self.key_address, data)

    def send_plaintext(self, plaintext : bytes):
        data = []
        for i in range(4):
            data.append(int.from_bytes(plaintext[4*i:4*i+4], byteorder='little'))
        self.write_data(self.plaintext_address, data)

    def run(self):
        stat = self.get_status()
        if stat["ap_idle"]:
            self.write_data(self.control_address + self.CTRL_ADDRESS_MAP["control"], [self.CTRL_AP_START])
        else:
            raise RuntimeError("HLS IP is not idle")

    def read_ciphertext(self, byte_len : int = 8):
        stat = self.get_status()
        if not stat["ap_done"]:
            raise RuntimeError("HLS IP is not done")

        read_words = self.read_data(self.ciphertext_address, byte_len // 4)
        ct = b''
        for w in read_words:
            ct += w.to_bytes(4, byteorder='little')
        self.write_data(self.control_address + self.CTRL_ADDRESS_MAP["control"], [self.CTRL_AP_CONTINUE])
        return ct

    def isDone(self):
        stat = self.get_status()
        return stat["ap_done"]

class SakuraXShellExampleAES128BitHLS(SakuraXShellAES):
    def getName(self):
        return "Sakura-X Shell AES128 Bit HLS Example"

    def getControl(self, **kwargs) -> SakuraXShellControlBase:
        return SakuraXShellAES128BitHLSControl(self.ser, **kwargs)


