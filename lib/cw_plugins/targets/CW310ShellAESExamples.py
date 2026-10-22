###
#   Copyright (C) 2026 University of Tsukuba
#   
#   File:          /lib/cw_plugins/targets/CW310ShellAESExamples.py
#   Project:       chipwhisperer-enhanced-plugins
#   Author:        Takuya Kojima in University of Tsukuba (tkojima@lila.cs.tsukuba.ac.jp)
#   Created Date:  21-10-2026 00:33:56
#   Last Modified: 22-10-2026 16:06:29
###


from .CW310Shell import CW310ShellBase
from .AES128RTLMixin import AES128RTLMixin
from .AES128HLSMixin import AES128HLSMixin


class CW310ShellExampleAES128BitRTL(AES128RTLMixin, CW310ShellBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def getPlatformName():
        return "cw310"

    @staticmethod
    def isBsfileSupported():
        return True

    @staticmethod
    def hasUsbTrigger():
        return True


class CW310ShellExampleAES128BitHLS(AES128HLSMixin, CW310ShellBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def getPlatformName():
        return "cw310"

    @staticmethod
    def isBsfileSupported():
        return True

