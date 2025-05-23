###
#   Copyright (C) 2025 The University of Tokyo
#   
#   File:          /pytest/test_fastcpa.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  14-05-2025 07:47:08
#   Last Modified: 23-05-2025 18:22:18
###

import chipwhisperer as cw
import chipwhisperer.analyzer as cwa
from cw_plugins.analyzer.attacks.cpa_algorithms.fast_progressive import FastCPAProgressive, FastCPAProgressiveCuda,  FastCPAProgressiveOpenCL, FastCPAProgressiveOpenCLFP32, FastCPAProgressiveCudaFP32

import pytest

import pickle
import math

@pytest.fixture
def test_data(scope="module"):
    # Open test project
    proj = cw.open_project("./cwlite_trace.cwp")

    # Set the leakage model
    model = cwa.leakage_models.sbox_output

    # get data for validation obtained from the original implementation
    fname = proj.getDataFilepath("cw_best_geuss.pkl")["abs"]
    with open(fname, "rb") as f:
        ref_result = pickle.load(f)

    return proj, model, ref_result

def run_attack(attack, ref_result):
    """
    Run the attack and return the result.
    """
    result = attack.run(update_interval=100)
    guesses = result.best_guesses()
    # check key & correlation
    for i in range(16):
        assert guesses[i]["guess"] == ref_result[i]["guess"]
        assert math.isclose(guesses[i]["correlation"], ref_result[i]["correlation"],
                            rel_tol=1e-15, abs_tol=0.0)


def test_fastcpa_cpu(test_data):
    proj, model, ref_result = test_data
    attack = cwa.cpa(proj, model, FastCPAProgressive)
    run_attack(attack, ref_result)


def test_fastcpa_cuda(test_data):
    proj, model, ref_result = test_data
    try:
        attack = cwa.cpa(proj, model, FastCPAProgressiveCuda)
        run_attack(attack, ref_result)
    except ModuleNotFoundError:
        pytest.skip("Skipping test due to missing Cuda implementation")

def test_fastcpa_cuda_fp32(test_data):
    proj, model, ref_result = test_data
    try:
        attack = cwa.cpa(proj, model, FastCPAProgressiveCudaFP32)
        run_attack(attack, ref_result)
    except ModuleNotFoundError:
        pytest.skip("Skipping test due to missing Cuda implementation")


def test_fastcpa_opencl(test_data):
    proj, model, ref_result = test_data
    try:
        attack = cwa.cpa(proj, model, FastCPAProgressiveOpenCL)
        run_attack(attack, ref_result)
    except ModuleNotFoundError:
        pytest.skip("Skipping test due to missing OpenCL implementation")


def test_fastcpa_opencl_fp32(test_data):
    proj, model, ref_result = test_data
    try:
        attack = cwa.cpa(proj, model, FastCPAProgressiveOpenCLFP32)
        run_attack(attack, ref_result)
    except ModuleNotFoundError:
        pytest.skip("Skipping test due to missing OpenCLFP32 implementation")