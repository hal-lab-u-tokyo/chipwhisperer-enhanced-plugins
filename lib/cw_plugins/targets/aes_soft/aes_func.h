/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /lib/cw_plugins/targets/aes_soft/aes_func.h
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  21-07-2024 20:22:44
*    Last Modified: 16-05-2025 11:56:05
*/

#include <stdint.h>
#include "aes_table.h"
#define BLOCKLEN 16

#define POLY 0x1B  // irreducible polynomial of AES: x^8 + x^4 + x^3 + x + 1 (b0001_1011)

// put the following arrays in .bss section instead of stack to prevent randomly happened cache miss
static uint8_t __attribute__((aligned(32))) state[BLOCKLEN];
static uint8_t __attribute__((aligned(32))) round_key[BLOCKLEN];
static uint8_t __attribute__((aligned(32))) next_state[BLOCKLEN];

// used only for masking implementation
#ifdef MASKING
static uint8_t __attribute__ ((aligned(256))) masked_sbox_table[256];
static uint8_t __attribute__ ((aligned(32))) masked_round_key[BLOCKLEN];

static inline uint8_t masked_sbox(uint8_t a)
{
	return masked_sbox_table[a];
}

typedef struct {
	uint32_t m; // {m,m,m,m}
	uint32_t m_; // {m',m',m',m'}
	uint32_t m_x;  // {m1,m2,m3,m4}
	uint32_t m_x_; // {m1',m2',m3',m4'}
} mask_t;

void create_mask(mask_t *mask);
void remasking(uint8_t *state, mask_t *mask);
void masking_sbox(mask_t *mask);
void masked_sub_bytes(uint8_t *state);
void masking_round_key(uint8_t *round_key, uint8_t *masked_round_key, mask_t  *mask);

#endif // MASKING

#define UNROLL_SHIFT_ROWS
#define USE_TABLE

#ifdef USE_TABLE
static inline uint8_t gf_x2(uint8_t a)
{
	return mul_2[a];
}

static inline uint8_t gf_x3(uint8_t a)
{
	return mul_3[a];
}

#else

static inline uint8_t gf_x2(uint8_t a)
{
	return (a << 1) ^ ((a & 0x80) ? POLY : 0);
}

static inline uint8_t gf_x3(uint8_t a)
{
	return gf_x2(a) ^ a;
}

#endif

static inline uint8_t sbox(uint8_t a)
{
	return sbox_table[a];
}

static inline void mix_single_column(uint8_t *column, uint8_t *next_column)
{
	next_column[0] = gf_x2(column[0]) ^ gf_x3(column[1]) ^ column[2] ^ column[3];
	next_column[1] = column[0] ^ gf_x2(column[1]) ^ gf_x3(column[2]) ^ column[3];
	next_column[2] = column[0] ^ column[1] ^ gf_x2(column[2]) ^ gf_x3(column[3]);
	next_column[3] = gf_x3(column[0]) ^ column[1] ^ column[2] ^ gf_x2(column[3]);
}


void add_round_key(uint8_t *state, uint8_t *round_key);
void sub_bytes(uint8_t *state);
#ifdef MASKING
#define SHIFT_ROWS_ARGS uint8_t *state, mask_t *mask
#else
#define SHIFT_ROWS_ARGS uint8_t *state
#endif
void shift_rows(SHIFT_ROWS_ARGS);
void mix_columns(uint8_t *state);
void key_expansion(uint8_t *key, int round);

void aes_encrypt(const uint8_t* key, const uint8_t* plaintext, uint8_t* ciphertext);


