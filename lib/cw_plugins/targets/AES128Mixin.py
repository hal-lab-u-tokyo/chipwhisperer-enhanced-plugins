###
#   Copyright (C) 2026 University of Tsukuba
#   
#   File:          /lib/cw_plugins/targets/AES128Mixin.py
#   Project:       chipwhisperer-enhanced-plugins
#   Author:        Takuya Kojima in University of Tsukuba (tkojima@lila.cs.tsukuba.ac.jp)
#   Created Date:  22-10-2026 17:54:23
#   Last Modified: 22-10-2026 17:54:26
###



from Crypto.Cipher import AES

class AES128Mixin:
    def __init__(self):
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

    def registerEncryptionKey(self, key):
        self.last_key = key
        self.cipher = AES.new(bytes(key), AES.MODE_ECB)

    def registerInput(self, inputtext):
        self.input = inputtext

