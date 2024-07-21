/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /lib/cw_plugins/targets/aes_soft/aes.c
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  21-07-2024 20:22:27
*    Last Modified: 21-07-2024 20:22:28
*/


#include "aes_func.h"
#include <minilib.h>
#include <serial.h>

typedef int bool;

#define TRUE (1)
#define FALSE (0)

#define CMD_SET_KEY 		0x11
#define CMD_SET_PLAINTEXT	0x12
#define CMD_ENCRYPT			0x13
#define CMD_GET_CIPHERTEXT	0x14

#define RESPONSE_OK			0x00
#define RESPONSE_ERROR		0x01


int main()
{
	uint8_t key[BLOCKLEN];
	uint8_t plaintext[BLOCKLEN];
	uint8_t ciphertext[BLOCKLEN];

	// init arrays
	for (int i = 0; i < BLOCKLEN; i++) {
		key[i] = 0;
		plaintext[i] = 0;
		ciphertext[i] = 0;
	}
	// recv buffer clear
	volatile unsigned char dummy;
	while (serial_available() > 0) {
		dummy = serial_recv();
	}

	bool key_ready = FALSE;

	led_all_off();
	pinHeaderWrite(0, 0);
	int pos = 0;

	while (1) {
		if (serial_available() > 0) {
			unsigned char cmd = serial_recv();
			switch (cmd) {
				case CMD_SET_KEY:
					pos = 0;
					while (pos < BLOCKLEN) {
						if (serial_available() > 0) {
							key[pos] = serial_recv();
							pos++;
						}
					}
					key_ready = TRUE;
					led_on(0);
					break;
				case CMD_SET_PLAINTEXT:
					pos = 0;
					while (pos < BLOCKLEN) {
						if (serial_available() > 0){
							plaintext[pos] = serial_recv();
							pos++;
						}
					}
					break;
				case CMD_ENCRYPT:
					if (key_ready) {
						aes_encrypt(key, plaintext, ciphertext);
						serial_send(RESPONSE_OK);
					} else {
						// key is not set
						serial_send(RESPONSE_ERROR);
					}
					break;
				case CMD_GET_CIPHERTEXT:
					for (int i = 0; i < BLOCKLEN; i++) {
						serial_send(ciphertext[i]);
					}
					break;
				default:
					break;
			}
		}
	}

	return 0;
}
