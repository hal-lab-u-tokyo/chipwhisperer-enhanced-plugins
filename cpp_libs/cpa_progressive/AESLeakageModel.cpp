#include "AESLeakageModel.hpp"


using namespace AESLeakageModel;
// int ModelBase::leakage(py::array_t<uint8_t> &py_plaintext, py::array_t<uint8_t> &py_ciphertext, py::array_t<uint8_t> &py_key, int byte_index)
// {
// 	return hamming_weight[leakage_impl((uint8_t *)py_plaintext.request().ptr, (uint8_t *)py_ciphertext.request().ptr, (uint8_t *)py_key.request().ptr, byte_index)];
// }

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