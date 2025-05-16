###
#   Copyright (C) 2025 The University of Tokyo
#   
#   File:          /pytest/test_fastcpa.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  14-05-2025 07:47:08
#   Last Modified: 14-05-2025 07:47:12
###

import chipwhisperer as cw
import chipwhisperer.analyzer as cwa
from cw_plugins.analyzer.attacks.cpa_algorithms.fast_progressive import FastCPAProgressive, FastCPAProgressiveCuda,  FastCPAProgressiveOpenCL, FastCPAProgressiveOpenCLFP32

import pytest

@pytest.fixture
def test_data(scope="module"):
    # Open test project
    proj = cw.open_project("./cwlite_trace.cwp")

    # Set the leakage model
    model = cwa.leakage_models.sbox_output

    # correct key
    key = model.process_known_key(proj.keys[0])

    return proj, model, key

def test_fastcpa_cpu(test_data):
    proj, model, key = test_data
    attack = cwa.cpa(proj, model, FastCPAProgressive)
    result = attack.run()
    result.set_known_key(key)
    result.calc_PGE(0)
    assert (result.key_guess() == key).all()


def test_fastcpa_cuda(test_data):
    proj, model, key = test_data
    try:
        print("Running test with Cuda implementation")
        attack = cwa.cpa(proj, model, FastCPAProgressiveCuda)
        result = attack.run()
    except ModuleNotFoundError:
        pytest.skip("Skipping test due to missing Cuda implementation")
    result.set_known_key(key)
    result.calc_PGE(0)
    assert (result.key_guess() == key).all()

def test_fastcpa_opencl(test_data):
    proj, model, key = test_data
    try:
        attack = cwa.cpa(proj, model, FastCPAProgressiveOpenCL)
        result = attack.run()
    except ModuleNotFoundError:
        pytest.skip("Skipping test due to missing OpenCL implementation")
    result.set_known_key(key)
    result.calc_PGE(0)
    assert (result.key_guess() == key).all()

def test_fastcpa_opencl_fp32(test_data):
    proj, model, key = test_data
    try:
        attack = cwa.cpa(proj, model, FastCPAProgressiveOpenCLFP32)
        result = attack.run()
    except ModuleNotFoundError:
        pytest.skip("Skipping test due to missing OpenCLFP32 implementation")
    result.set_known_key(key)
    result.calc_PGE(0)
    assert (result.key_guess() == key).all()