###
#   Copyright (C) 2025 The University of Tokyo
#   
#   File:          /lib/cw_plugins/analyzer/attacks/cpa_algorithms/models.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  30-05-2025 08:03:04
#   Last Modified: 30-05-2025 08:03:08
###


from chipwhisperer.analyzer.attacks.models.AES128_8bit import *
from . import model_kernel

from chipwhisperer.analyzer.attacks.models.AES128_8bit import AESLeakageHelper

class PlaintextKeyXORDiff(AESLeakageHelper):
    def __init__(self):
        self.last_state = [0]*16
    def leakage(self, pt, ct, key, bnum):
        st1 = self.last_state[bnum]
        sbox_in = pt[bnum] ^ key[bnum]
        sbox_out = self.sbox(sbox_in)
        self.last_state[bnum] = ct[bnum]
        return st1 ^ sbox_out


model_dict = {
    SBox_output: model_kernel.SBoxOutput,
    SBoxInOutDiff: model_kernel.SBoxInOutDiff,
    LastroundStateDiff: model_kernel.LastRoundStateDiff,
    LastroundHW: model_kernel.LastRoundState,
    LastroundStateDiffAlternate: model_kernel.LastRoundStateDiffAlternate,
    PtKey_XOR: model_kernel.PlaintextKeyXOR,
    PlaintextKeyXORDiff: model_kernel.PlaintextKeyXORDiff
}

def get_c_model(model):
    return model_dict[model.model]()

