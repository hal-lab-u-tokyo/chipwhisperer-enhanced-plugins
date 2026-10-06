###
#   Copyright (C) 2024 The University of Tokyo
#   
#   File:          /lib/cw_plugins/targets/SakuraXVexRISCVAESExample.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  15-07-2024 19:24:25
#   Last Modified: 29-05-2025 07:18:11
###


from .SakuraShellAESExamples import SakuraXShellAES
from .SakuraXVexRISCV import SakuraXVexRISCVControlBase

import os

UNMASKED_PROGRAM = os.path.join(os.path.dirname(__file__), "aes_soft/sakura-x/aes_unmasked.elf")
MASKED_PROGRAM = os.path.join(os.path.dirname(__file__), "aes_soft/sakura-x/aes_masked.elf")

class SakuraXVexRISCVControlAES128bit(SakuraXVexRISCVControlBase):
    CMD_SET_KEY 		= 0x11
    CMD_SET_PLAINTEXT	= 0x12
    CMD_ENCRYPT			= 0x13
    CMD_GET_CIPHERTEXT	= 0x14
    CMD_GET_DEBUG		= 0x15

    def __init__(self, ser, masked = False, **kwargs):
        if masked:
            print("Masked AES program is selected")
            program = MASKED_PROGRAM
        else:
            print("Unprotected AES program is selected")
            program = UNMASKED_PROGRAM
        super().__init__(ser, program, **kwargs)


    def send_key(self, key : bytes):
        buf = b""
        buf += self.CMD_SET_KEY.to_bytes(1, 'big')
        buf += key
        self.send_bytes(buf)

    def send_plaintext(self, plaintext : bytes):
        buf = b""
        buf += self.CMD_SET_PLAINTEXT.to_bytes(1, 'big')
        buf += plaintext
        self.send_bytes(buf)

    def run(self):
        buf = b""
        buf += self.CMD_ENCRYPT.to_bytes(1, 'big')
        self.send_bytes(buf)
        # wait response
        stat = self.recv_bytes(1)
        if stat[0] != 0:
            print(stat)
            print(self.read_ciphertext(16))
            print()
            raise RuntimeError("Encryption failed")

    def read_ciphertext(self, byte_len : int = 8):
            buf = b""
            buf += self.CMD_GET_CIPHERTEXT.to_bytes(1, 'big')
            self.send_bytes(buf)
            return self.recv_bytes(16)

class SakuraXVexRISCVAESExample(SakuraXShellAES):
    def getControl(self, **kwargs) -> SakuraXVexRISCVControlAES128bit:
        return SakuraXVexRISCVControlAES128bit(self.ser, **kwargs)

    def getName(self):
        return "Sakura-X VexRISCV AES Example"