###
#   Copyright (C) 2026 University of Tsukuba
#   
#   File:          /lib/cw_plugins/targets/VexRiscvAESExamples.py
#   Project:       chipwhisperer-enhanced-plugins
#   Author:        Takuya Kojima in University of Tsukuba (tkojima@lila.cs.tsukuba.ac.jp)
#   Created Date:  15-07-2024 19:24:25
#   Last Modified: 22-10-2026 18:31:44
###



from cw_plugins.targets.CW305Shell import CW305ShellBase

from .SakuraXShell import SakuraXShellBase
from .AES128Mixin import AES128Mixin
from .VexRiscvMixin import VexRiscvMixin

import numpy as np

from pathlib import Path


class VexRiscvAESExample(AES128Mixin, VexRiscvMixin):
    CMD_SET_KEY 		= 0x11
    CMD_SET_PLAINTEXT	= 0x12
    CMD_ENCRYPT			= 0x13
    CMD_GET_CIPHERTEXT	= 0x14
    CMD_GET_DEBUG		= 0x15

    def __init__(self):
        super().__init__()
        self.program = None

    def loadEncryptionKey(self, key : bytes):
        self.registerEncryptionKey(key)
        buf = b""
        buf += self.CMD_SET_KEY.to_bytes(1, 'big')
        buf += key
        self.send_bytes(buf)

    def loadInput(self, plaintext : bytes):
        self.registerInput(plaintext)
        buf = b""
        buf += self.CMD_SET_PLAINTEXT.to_bytes(1, 'big')
        buf += plaintext
        self.send_bytes(buf)

    def go(self):
        buf = b""
        buf += self.CMD_ENCRYPT.to_bytes(1, 'big')
        self.send_bytes(buf)
        # wait response
        stat = self.recv_bytes(1)
        if stat[0] != 0:
            self.debug_print(stat)
            self.debug_print(self.readOutput(16))
            self.debug_print("\n")
            raise RuntimeError("Encryption failed")

    def readOutput(self):
            buf = b""
            buf += self.CMD_GET_CIPHERTEXT.to_bytes(1, 'big')
            self.send_bytes(buf)
            return np.frombuffer(self.recv_bytes(16), dtype=np.uint8).copy()



class SakuraXVexRiscvAESExample(VexRiscvAESExample, SakuraXShellBase):
    @staticmethod
    def isBsfileSupported():
        return False

    @staticmethod
    def getPlatformName():
        return "sakura-x"
    
    def _con(self, scope = None, masked = False, **kwargs):
        filename = "aes_masked.elf" if masked else "aes_unmasked.elf"
        self.program = Path(__file__).parent / "aes_soft" / self.getPlatformName() / filename
        super()._con(scope, program=self.program, **kwargs)


        if masked:
            self.debug_print("Masked AES program is selected")
        else:
            self.debug_print("Unprotected AES program is selected")



class CW305VexRiscvAESExample(VexRiscvAESExample, CW305ShellBase):
    @staticmethod
    def isBsfileSupported():
        return True

    @staticmethod
    def getPlatformName():
        return "cw305"
    
    def _con(self, scope = None, masked = False, **kwargs):
        filename = "aes_masked.elf" if masked else "aes_unmasked.elf"
        self.program = Path(__file__).parent / "aes_soft" / self.getPlatformName() / filename
        super()._con(scope, program=self.program, **kwargs)


        if masked:
            self.debug_print("Masked AES program is selected")
        else:
            self.debug_print("Unprotected AES program is selected")

