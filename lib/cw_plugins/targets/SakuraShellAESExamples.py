###
#   Copyright (C) 2024 The University of Tokyo
#   
#   File:          /lib/cw_plugins/targets/SakuraShellAESExamples.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  13-07-2024 15:38:26
#   Last Modified: 15-07-2024 19:23:57
###

from .SakuraXShell import SakuraXShellBase, SakuraXShellControlBase
from Crypto.Cipher import AES

from abc import ABCMeta

class SakuraXShellAES(SakuraXShellBase, metaclass=ABCMeta):

    def __init__(self, ):
        super().__init__()
        self.cipher = None
        self.last_key = [0 for _ in range(16)]
        self.key_size = 16
        self.input_size = 16
        self.output_size = 16
        self.input = bytes()

    def getkeySize(self):
        return self.key_size

    def getInputSize(self):
        return self.input_size

    def getOutputSize(self):
        return self.output_size

    def getExpected(self):
        ct = self.cipher.encrypt(bytes(self.input))
        ct = bytearray(ct)
        return ct

    def loadEncryptionKey(self, key):
        self.key_size = len(key)
        self.ctrl.send_key(bytes(key))
        self.last_key = key
        self.cipher = AES.new(bytes(key), AES.MODE_ECB)

    def loadInput(self, inputtext):
        self.input_size = len(inputtext)
        self.output_size = len(inputtext)
        self.input = inputtext
        self.ctrl.send_plaintext(bytes(inputtext))



# Example Implementations for Sakura-X Shell

class SakuraXShellAES128BitRTLControl(SakuraXShellControlBase):
    ADDRESS_MAP = {
        "key": 0x0,
        "plaintext": 0x10,
        "ciphertext": 0x20,
        "control": 0x30,
    }
    KEY_READY_BIT = 0x2
    PT_READY_BIT = 0x1
    def __init__(self, ser, address_base = 0x8000_0000, **kwargs):
        super().__init__(ser)
        self.reset_command()
        self.address_base = address_base

    def send_key(self, key : bytes):
        key_words = [int.from_bytes(key[4*i:4*i+4], byteorder='big') for i in range(4)]
        self.write_data(self.address_base + self.ADDRESS_MAP["key"], key_words[::-1])

    def send_plaintext(self, plaintext : bytes):
        pt_words = [int.from_bytes(plaintext[4*i:4*i+4], byteorder='big') for i in range(4)]
        self.write_data(self.address_base + self.ADDRESS_MAP["plaintext"], pt_words[::-1])

    def run(self):
        self.write_data(self.address_base + self.ADDRESS_MAP["control"], \
                         [self.KEY_READY_BIT])
        self.write_data(self.address_base + self.ADDRESS_MAP["control"], \
                            [self.PT_READY_BIT])

    def read_ciphertext(self, byte_len : int = 8):
        read_words = self.read_data(self.address_base + self.ADDRESS_MAP["ciphertext"], byte_len // 4)
        ct = b''
        for w in read_words[::-1]:
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

    def __init__(self, ser, control_address = 0x8000_0000, bram_address = 0xC000_0000, **kwargs):
        super().__init__(ser)
        self.reset_command()
        self.control_address = control_address
        self.key_address = bram_address
        self.plaintext_address = bram_address + 0x10
        self.ciphertext_address = bram_address + 0x20

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


