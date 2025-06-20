/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /lib/cw_plugins/targets/aes_soft/aes_func.c
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  21-07-2024 20:22:55
*    Last Modified: 17-05-2025 02:36:04
*/

#include "aes_func.h"
#include <minilib.h>


void add_round_key(uint8_t *state, uint8_t *round_key)
{
	for (int i = 0; i < (BLOCKLEN / 4); i++) {
		((uint32_t*)state)[i] ^= ((uint32_t*)round_key)[i];
	}
}

void sub_bytes(uint8_t *state)
{
	for (int i = 0; i < BLOCKLEN; i++) {
		state[i] = sbox(state[i]);
	}
}


void shift_rows(SHIFT_ROWS_ARGS)
{

#ifdef UNROLL_SHIFT_ROWS

	next_state[0] = state[0];
	next_state[1] = state[5];
	next_state[2] = state[10];
	next_state[3] = state[15];

	next_state[4] = state[4];
	next_state[5] = state[9];
	next_state[6] = state[14];
	next_state[7] = state[3];

	next_state[8] = state[8];
	next_state[9] = state[13];
	next_state[10] = state[2];
	next_state[11] = state[7];

	next_state[12] = state[12];
	next_state[13] = state[1];
	next_state[14] = state[6];
	next_state[15] = state[11];

#else
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			int idx = (i + j) % 4 * 4 + j;
			next_state[4 * i + j] = state[idx];

		}
	}
#endif
	for (int i = 0; i < (BLOCKLEN / 4); i++) {
		((uint32_t*)state)[i] = ((uint32_t*)next_state)[i];
	}

}

void mix_columns(uint8_t *state)
{
	uint8_t next_state[BLOCKLEN];
	for (int i = 0; i < 4; i++) {
		mix_single_column(state + i * 4, next_state + i * 4);
	}
	for (int i = 0; i < (BLOCKLEN / 4); i++) {
		((uint32_t*)state)[i] = ((uint32_t*)next_state)[i];
	}
}

void key_expansion(uint8_t *key, int round)
{
	uint8_t rcon = rcon_table[round];
	uint8_t sb_key[4];
	sb_key[0] = sbox(key[12]);
	sb_key[1] = sbox(key[13]);
	sb_key[2] = sbox(key[14]);
	sb_key[3] = sbox(key[15]);

	__attribute__((aligned(32))) uint8_t rot_word[4];
	rot_word[0] = sb_key[1] ^ rcon;
	rot_word[1] = sb_key[2];
	rot_word[2] = sb_key[3];
	rot_word[3] = sb_key[0];


	uint32_t *key32 = (uint32_t *)key;
	uint32_t *rot_word32 = (uint32_t *)rot_word;

	key32[0] = key32[0] ^ *rot_word32;
	key32[1] = key32[1] ^ key32[0];
	key32[2] = key32[2] ^ key32[1];
	key32[3] = key32[3] ^ key32[2];

}

#ifdef MASKING
void create_mask(mask_t *mask)
{
	uint8_t rnd;
	rnd = rand() & 0xff;
	mask->m = (rnd << 24) | (rnd << 16) | (rnd << 8) | rnd;
	rnd = rand() & 0xff;
	mask->m_ = (rnd << 24) | (rnd << 16) | (rnd << 8) | rnd;
	mask->m_x = rand();
	mix_single_column((uint8_t*)(&mask->m_x), (uint8_t*)(&mask->m_x_));
}

void masking_sbox(mask_t *mask)
{
	uint8_t m = mask->m & 0xff;
	uint8_t m_ = mask->m_ & 0xff;
	for (int i = 0; i < 256; i++) {
		masked_sbox_table[i] = sbox(i ^ m) ^ m_;
	}
}


// #pragma GCC push_options
// #pragma GCC optimize ("O0")

// __attribute__((optnone))
__attribute__((noinline))
void remasking(uint8_t *state, mask_t *mask)
{
#if defined(LEAKY_IMPL0)
	// vulnerability 1: reuse of register
	uint32_t mask_value = mask->m_x ^ mask->m_;
	for (int i = 0; i < 4; i++) {
		((uint32_t*)state)[i] ^= mask_value;
	}
#elif defined(LEALY_IMPL1)
	register uint32_t m_ = mask->m_;
	// guaruntee that m_ is loaded first
	__asm__ volatile ("":::"memory");
	register uint32_t m_x = mask->m_x;
	for (int i = 0; i < 4; i++) {
		((uint32_t*)state)[i] ^= m_x ^ m_;
	}
#elif defined(LEAKY_IMPL2)
	// separate XORing but due to memory remnant leakage, it is still vulnerable
	uint32_t m_x = mask->m_x;
	for (int i = 0; i < 4; i++) {
		((uint32_t*)state)[i] ^= m_x;
	}
	// The last load data is state[12,1,6,11] masked with m', then, m' is loaded
	// So, the hamming distance is equal to the hamming weight of plain state[12,1,6,11]
	uint32_t m_ = mask->m_;
	for (int i = 0; i < 4; i++) {
		((uint32_t*)state)[i] ^= m_;
	}
#else
	uint32_t m_x = mask->m_x;
#ifndef LEAKY_IMPL3
	// clang -O2 by default unrolls the loop, however, it might leak the intermediate value
	#pragma nounroll
#endif
	for (int i = 0; i < 4; i++) {
		((uint32_t*)state)[i] ^= m_x;
	}
	// inserting dummy mem load to clear sensitive data (last load is state[12,1,6,11]) in hidden registers
	volatile uint32_t dummy;
	*(volatile uint32_t*)&dummy;
	__asm__ volatile ("":::"memory");
	uint32_t m_ = mask->m_;
#ifndef LEAKY_IMPL3
	// clang -O2 by default unrolls the loop, however, it might leak the intermediate value
	#pragma nounroll
#endif
	for (int i = 0; i < 4; i++) {
		((uint32_t*)state)[i] ^= m_;
	}
#endif

}

// #pragma GCC pop_options


void masked_sub_bytes(uint8_t *state)
{
	for (int i = 0; i < BLOCKLEN; i++) {
		state[i] = masked_sbox(state[i]);
	}
}

void masking_round_key(uint8_t *round_key, uint8_t *masked_round_key, mask_t  *mask)
{
	for (int i = 0; i < 4; i++) {
		((uint32_t*)masked_round_key)[i] = ((uint32_t*)round_key)[i] ^ mask->m_x_ ^ mask->m;
	}
}

void aes_encrypt(const uint8_t* key, const uint8_t* plaintext, uint8_t* ciphertext)
{

	mask_t mask;
	create_mask(&mask);
	masking_sbox(&mask);

	pinHeaderWrite(0, 1);

	// copy
	for (int i = 0; i < 4; i++) {
		((uint32_t*)state)[i] = ((uint32_t*)plaintext)[i] ^ mask.m_x_;
		((uint32_t*)round_key)[i] = ((uint32_t*)key)[i];
	}

	masking_round_key(round_key, masked_round_key, &mask);
	add_round_key(state, masked_round_key);

	for (int round = 0; round < 9; round++) {
		masked_sub_bytes(state);
		shift_rows(state, &mask);
		remasking(state, &mask);
		mix_columns(state);
		key_expansion(round_key, round);
		masking_round_key(round_key, masked_round_key, &mask);
		add_round_key(state, masked_round_key);
		if (round == 0) {
			pinHeaderWrite(0, 0);
		}
	}
	// final round
	masked_sub_bytes(state);
	shift_rows(state, &mask);
	key_expansion(round_key, 9);
	masking_round_key(round_key, masked_round_key, &mask);
	add_round_key(state, masked_round_key);

	// store ciphertext while eliminating mask
	for (int i = 0; i < 4; i++) {
		((uint32_t*)ciphertext)[i] = ((uint32_t*)state)[i] ^ mask.m ^ mask.m_ ^ mask.m_x_;
	}
}

#else
void aes_encrypt(const uint8_t* key, const uint8_t* plaintext, uint8_t* ciphertext) {

	pinHeaderWrite(0, 1);

	// copy
	for (int i = 0; i < 4; i++) {
		((uint32_t*)state)[i] = ((uint32_t*)plaintext)[i];
		((uint32_t*)round_key)[i] = ((uint32_t*)key)[i];
	}

	add_round_key(state, round_key);

	for (int round = 0; round < 9; round++) {
		sub_bytes(state);
		shift_rows(state);
		mix_columns(state);
		key_expansion(round_key, round);
		add_round_key(state, round_key);

		if (round == 0) {
			pinHeaderWrite(0, 0);
		}
	}
	// final round
	sub_bytes(state);
	shift_rows(state);
	key_expansion(round_key, 9);
	add_round_key(state, round_key);
	for (int i = 0; i < 4; i++) {
		((uint32_t*)ciphertext)[i] = ((uint32_t*)state)[i];
	}
}
#endif
