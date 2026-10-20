###
#   Copyright (C) 2026 University of Tsukuba
#   
#   File:          /lib/cw_plugins/targets/CW305ShellAESExamples.py
#   Project:       chipwhisperer-enhanced-plugins
#   Author:        Takuya Kojima in University of Tsukuba (tkojima@lila.cs.tsukuba.ac.jp)
#   Created Date:  21-10-2026 00:33:56
#   Last Modified: 21-10-2026 00:33:59
###


from .CW305Shell import CW305ShellBase
from .AES128RTLMixin import AES128RTLMixin
from .AES128HLSMixin import AES128HLSMixin

from Crypto.Cipher import AES

class CW305ShellExampleAES128BitRTL(AES128RTLMixin, CW305ShellBase):
    def __init__(self):
        super().__init__()
    @staticmethod
    def getPlatformName():
        return "CW305"

class CW305ShellExampleAES128BitHLS(AES128HLSMixin, CW305ShellBase):
    def __init__(self):
        super().__init__()
    @staticmethod
    def getPlatformName():
        return "CW305"
