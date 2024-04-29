# Target hardware implementation

## [ESP32_AES128](../hardware/ESP32_AES128/)

PlatformIO project for AES-128 implementation on ESP32 is provided.
Please change macro in `main.cpp` as you like.
* CPU_FREQ: CPU frequency in MHz
* LED_PIN: GPIO pin number for LED
* TRIGGER_PIN: GPIO pin number for trigger signal

If AES encryption is ready on ESP32, the LED will blink.

The trigger signal is asserted when the encryption starts and deasserted when the encryption ends.

To communicate with the ESP32, the UART-USB converter is required.

