/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /cpp_libs/cpa_progressive/AESLeakageModel.cpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  30-01-2024 12:30:57
*    Last Modified: 30-01-2024 12:31:15
*/


#include "AESLeakageModel.hpp"

using namespace AESLeakageModel;


int SBoxOutput::leakage_impl(uint8_t * plaintext, uint8_t * ciphertext, uint8_t * key, int byte_index)
{
	return sbox(plaintext[byte_index] ^ key[byte_index]);
}

int SBoxInOutDiff::leakage_impl(uint8_t * plaintext, uint8_t * ciphertext, uint8_t * key, int byte_index)
{
	auto st1 = plaintext[byte_index] ^ key[byte_index];
	auto st2 = sbox(st1);
	return st1 ^ st2;
}


int LastRoundStateDiff::leakage_impl(uint8_t * plaintext, uint8_t * ciphertext, uint8_t * key, int byte_index)
{
	auto st10 = ciphertext[invshift_undo[byte_index]];
	auto st9 = inv_sbox(ciphertext[byte_index] ^ key[byte_index]);
	return (st10 ^ st9);
}