#include <Arduino.h>
#include <WiFi.h>

#define BLOCKLEN 16 // AES 8x16=128bit
#define CPU_FREQ 160 // MHz

unsigned char key[BLOCKLEN];
unsigned char plaintext[BLOCKLEN];
unsigned char ciphertext[BLOCKLEN];
bool key_ready = false;

#define LED_PIN 25
#define TRIGGER_PIN 27

#define CMD_SET_KEY 		0x11
#define CMD_SET_PLAINTEXT	0x12
#define CMD_ENCRYPT			0x13
#define CMD_GET_CIPHERTEXT	0x14
#define CMD_DEBUG			0x15


#include <aes/esp_aes.h>
int encrypt()
{
	esp_aes_context ctx;
	esp_aes_init(&ctx);
	esp_aes_setkey(&ctx, key, BLOCKLEN * 8);
	esp_aes_crypt_ecb(&ctx, ESP_AES_ENCRYPT, plaintext, ciphertext);
	esp_aes_free(&ctx);
	return 0;
}


void setup() {
	// set CPU frequency
	setCpuFrequencyMhz(CPU_FREQ);

	// setup serial
	Serial.begin(115200);

	// setup LED
	pinMode(LED_PIN, OUTPUT);
	digitalWrite(LED_PIN, LOW);
	pinMode(TRIGGER_PIN, OUTPUT);
	digitalWrite(TRIGGER_PIN, LOW);

	// turn off wifi, bluetooth
	WiFi.mode(WIFI_OFF);
	btStop();

	// init arrays
	for (int i = 0; i < BLOCKLEN; i++) {
		key[i] = 0;
		plaintext[i] = 0;
		ciphertext[i] = 0;
	}

	// disgard serial buffer
	while (Serial.available() > 0) {
		Serial.read();
	}

	// run fist encrypt
	encrypt();
}

void loop() {
	if (Serial.available() > 0) {
		unsigned char cmd = Serial.read();
		switch (cmd) {
			case CMD_SET_KEY:
				Serial.readBytes(key, BLOCKLEN);
				key_ready = true;
				digitalWrite(LED_PIN, HIGH);
				break;
			case CMD_SET_PLAINTEXT:
				Serial.readBytes(plaintext, BLOCKLEN);
				break;
			case CMD_ENCRYPT:
				if (key_ready) {
					digitalWrite(TRIGGER_PIN, HIGH);
					unsigned char ret = encrypt();
					digitalWrite(TRIGGER_PIN, LOW);
					Serial.write(ret);
				}
				break;
			case CMD_GET_CIPHERTEXT:
				Serial.write(ciphertext, BLOCKLEN);
				break;
			case CMD_DEBUG:
				Serial.write(key, BLOCKLEN);
				Serial.write(plaintext, BLOCKLEN);
				Serial.write(ciphertext, BLOCKLEN);
				break;
			default:
				break;
		}
	}

}

