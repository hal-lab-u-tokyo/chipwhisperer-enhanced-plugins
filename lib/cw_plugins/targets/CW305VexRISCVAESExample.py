from .CW305VexRISCV import CW305VexRISCVBase
import os
from Crypto.Cipher import AES
import numpy as np

UNMASKED_PROGRAM = os.path.join(os.path.dirname(__file__), "aes_soft/aes_unmasked.elf")
MASKED_PROGRAM = os.path.join(os.path.dirname(__file__), "aes_soft/aes_masked.elf")

class CW305RISCVAES128bit(CW305VexRISCVBase):
    CMD_SET_KEY 		= 0x11
    CMD_SET_PLAINTEXT	= 0x12
    CMD_ENCRYPT			= 0x13
    CMD_GET_CIPHERTEXT	= 0x14
    CMD_GET_DEBUG		= 0x15

    def __init__(self):
        super().__init__()
        self.program = UNMASKED_PROGRAM
        self.cipher = None
        self.input = bytes()

    def _con(self, scope = None, masked = False, **kwargs):
        if masked:
            print("Masked AES program is selected")
            self.program = MASKED_PROGRAM
        else:
            print("Unprotected AES program is selected")
            self.program = UNMASKED_PROGRAM
        super()._con(scope, program = self.program, **kwargs)


    def textLen(self):
        return 16

    def keyLen(self):
        return 16

    def go(self):
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

    def getExpected(self):
        ct = self.cipher.encrypt(bytes(self.input))
        ct = bytearray(ct)
        return ct

    def loadEncryptionKey(self, key):
        self.last_key = key
        self.cipher = AES.new(bytes(key), AES.MODE_ECB)
        buf = b""
        buf += self.CMD_SET_KEY.to_bytes(1, 'big')
        buf += key
        self.send_bytes(buf)

    def loadInput(self, inputtext):
        self.input = inputtext
        buf = b""
        buf += self.CMD_SET_PLAINTEXT.to_bytes(1, 'big')
        buf += inputtext
        self.send_bytes(buf)

    def readOutput(self):
        buf = b""
        buf += self.CMD_GET_CIPHERTEXT.to_bytes(1, 'big')
        self.send_bytes(buf)
        ct =  self.recv_bytes(16)
        return np.frombuffer(ct, dtype=np.uint8)