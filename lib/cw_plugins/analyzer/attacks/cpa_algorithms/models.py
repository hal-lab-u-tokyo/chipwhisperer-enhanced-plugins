from chipwhisperer.analyzer.attacks.models.AES128_8bit import *
from . import cpa_kernel

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
    SBox_output: cpa_kernel.SBoxOutput,
    SBoxInOutDiff: cpa_kernel.SBoxInOutDiff,
    LastroundStateDiff: cpa_kernel.LastRoundStateDiff,
    LastroundHW: cpa_kernel.LastRoundState,
    LastroundStateDiffAlternate: cpa_kernel.LastRoundStateDiffAlternate,
    PtKey_XOR: cpa_kernel.PlaintextKeyXOR,
    PlaintextKeyXORDiff: cpa_kernel.PlaintextKeyXORDiff
}

def get_c_model(model):
    return model_dict[model.model]()

