# Target hardware implementation

[sample notebook](../notebooks/acquire_traces.ipynb) is available for acquiring traces from the target devices.

You can find a code snnipet for the target hardware implementation at the end of this document.

## [ESP32_AES128](../hardware/ESP32_AES128/)

PlatformIO project for AES-128 implementation on ESP32 is provided.
Please change macro in `main.cpp` as you like.
* CPU_FREQ: CPU frequency in MHz
* LED_PIN: GPIO pin number for LED
* TRIGGER_PIN: GPIO pin number for trigger signal

If encrpition key is set, the LED is turned on.

The trigger signal is asserted when the encryption starts and deasserted when the encryption ends.

To communicate with the ESP32, the UART-USB converter is required.


## SAKURA-X Shell
This is a our template of SAKURA-X board for easy evaluation of your cryptographic module.
For more information, please refer to its [repo](https://github.com/hal-lab-u-tokyo/sakura-x-shell).

This repository provides a driver template.

### [SakuraXShellBase](../lib//cw_plugins/targets/SakuraXShell.py)
This is an abstract class for Sakura-X Shell derived from `TargetTemplate` in Chipwhisperer.
If the following methods are implemented, the target can be used with `Chipwhisperer.capture_trace`.

* `getkeySize(self) -> Int`: return the size of the encryption key in bytes.
* `getInputSize(self) -> Int`: return the size of the plaintext in bytes.
* `getOutputSize(self) -> Int`: return the size of the ciphertext in bytes.
* `getControl(self, **kwargs) -> SakuraXShellControlBase`  
 return contoller object for target hardware on Sakura-X board (see below).
* `getExpected(self) -> bytes`: return the expected ciphertext.
* `set_key(self, key : bytes)`: set the encryption key.
* `loadInput(self, inputtext: bytes)`: load the plaintext.

### [SakuraXShellControlBase](../lib//cw_plugins/targets/SakuraXShell.py)
This is also an abstract class.
An instance of a derived class is an object to define the control of the target hardware on Sakura-X Shell, including key setting, sending plaintext, receiving ciphertext, and starting encryption.
The base class already defines communication methods with shell part of Sakura-X board.
Thus, the derived class at least needs to implement the following methods.

* `send_key(self, key : bytes)`  
set the encryption key. This is called by `set_key` in `SakuraXShellBase`.
* `send_plaintext(self, plaintext : bytes)`  
send the plaintext to the target. This is called by `loadInput` in `SakuraXShellBase`.
* `run(self)`  
start the encryption. This is called by `go` in `SakuraXShellBase`, which is called by `Chipwhisperer.capture_trace`.
* `read_ciphertext(self, byte_len : int = 8) -> bytes`  
return the ciphertext. 

The following methods are defined in the base class `SakuraXShellControlBase` and can be used in the derived class.

* `write_data(self, addr : int, data : Iterable[Int])`  
write data to hardware deisgn on Kintex-7 FPGA via shell controller on Spartan-6 FPGA.  
Note that each element of `data` is treated as **32-bit** data.

* `read_data(self, addr : int, length : int) -> List[Int]`
read data from hardware design on Kintex-7 FPGA via shell controller on Spartan-6 FPGA.
Not that the length is the number of **32-bit** data (word size).

* `reset_command(self)`  
Send reset signal to modules on the Kintex-7 FPGA.

### Examples
That repository contains implementations for two sample designs for AES-128 encryption provided at [sakura-x-shell repo](https://github.com/hal-lab-u-tokyo/sakura-x-shell)

Before using those, please configure the FPGAs on the board with the provided bitstreams or your built bitstreams.

1. RTL implementation of AES-128
To acquire traces from the RTL implementation, please use [`SakuraXShellExampleAES128BitRTL`](../lib//cw_plugins/targets/SakuraShellAESExamples.py) class.

The pin 1 of CN8 is used for the trigger signal, similar to SASEBO-GIII sample.


2. HLS implementation of AES-128
To acquire traces from the HLS implementation, please use [`SakuraXShellExampleAES128BitHLS`](../lib//cw_plugins/targets/SakuraShellAESExamples.py) class.

The pin 1 of CN8 is used for the trigger signal, similar to SASEBO-GIII sample.

## VexRiscv_SakuraX
VexRiscv is a 32-bit RISC-V CPU core and [VexRiscv_SakuraX](https://github.com/hal-lab-u-tokyo/VexRiscv_SakuraX) provides an implementation of VexRiscv on the Sakura-X board.

It is also designed based on the Sakura-X Shell, and [`SakuraXVexRISCVControlBase`](../lib//cw_plugins/targets/SakuraXVexRISCV.py) defines additional methods for controlling the CPU core, derived from `SakuraXShellControlBase`.
As described above, the derived class needs to implement the four methods `send_key`, `send_plaintext`, `run`, and `read_ciphertext`.

Its constructor requires the following arguments:
* ser (serial.Serial): serial object for communication with SAKURA-X board
* program (str): path to the elf binary file to be loaded to the CPU core

After instantiation, the CPU core boots with the program and starts running.

### Serial like communication with CPU core
The CPU core has a serial-like communication interface, which can be used for sending and receiving data.

The following methods are defined in `SakuraXVexRISCVControlBase` and can be used in the derived class.

* `send_bytes(self, data : bytes)`  
send byte data to the buffer that CPU core receives.

* `read_bytes(self, length : int) -> bytes`  
read byte data from the buffer that CPU core sends.

The following methods are available to check buffer status
* `is_send_buffer_full(self)`
* `is_send_buffer_empty(self)`
* `is_recv_buffer_full(self)`
* `is_recv_buffer_empty(self)`
* `get_send_buffer_bytes(self)`
* `get_recv_buffer_bytes(self)`

### AES example

This repository includes software implementation of AES-128 encryption for VexRiscv.
[`SakuraXVexRISCVAESExample`](../lib//cw_plugins/targets/SakuraXVexRISCVAESExample.py) is a derived class of `SakuraXShellBase`.

The pin 1 of CN8 is used for the trigger signal, similar to SASEBO-GIII sample.

## Code snippet for above examples without trace acquisition

```python
import chipwhisperer as cw
from cw_plugins.targets.ESP32 import ESP32 # <- ESP32 example
from cw_plugins.targets import SakuraXShellExampleAES128BitRTL # <- for RTL design example
from cw_plugins.targets import SakuraXShellExampleAES128BitHLS # <- for HLS design example
from cw_plugins.targets import SakuraXVexRISCVAESExample # <- for VexRiscv with software AES example

# instantiate the target hardware
target = SakuraXShellExampleAES128BitRTL()
scope = None # <- because oscilloscope is not used
target.con(scope, serial_port="/dev/ttyUSB0")

# key, plaintext generator in Chipwhisperer
ktp = cw.ktp.Basic()

# key loading
key = ktp.next_key()
print("key:", key)
target.loadEncryptionKey(key)

# plaintext loading
plaintext = ktp.next_text()
print("plaintext:", plaintext)
target.loadInput(plaintext)

# start encryption
target.go()

# wait for enough time or check the status of the target

# read ciphertext
ciphertext = cw.bytearray(target.readOutput())
print("ciphertext:", ciphertext)

expected = cw.bytearray(target.getExpected())
if ciphertext == expected:
	print("Success!")
else:
	print("Failed!")

```
