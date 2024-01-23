from chipwhisperer.analyzer.attacks.models.AES128_8bit import *
from . import cpa_kernel

model_dict = {
    SBox_output: cpa_kernel.SBoxOutput,
    SBoxInOutDiff: cpa_kernel.SBoxInOutDiff,
    LastroundStateDiff: cpa_kernel.LastRoundStateDiff,
}

def get_c_model(model):
    return model_dict[model.model]()