# Supported Target Hardware

The plugins support the following target hardware for acquiring traces and controlling the target devices.
* [SAKURA-X/SASEBO-GIII sample design](#sakura-x-sasebo-giii-sample-design)
* [ESP32](#esp32_aes128)
* [SAKURA-X Shell](#sakura-x-shell)
* [CW305 Shell](#cw305-shell)
* VexRiscv_SCA on the shells

All of the target hardware can be controlled with their corresponding target classes, which are derived from `TargetTemplate` in Chipwhisperer.
Therefore, it can be used in the same manner as the built-in target classes in Chipwhisperer like `cw.target`.

<!-- [sample notebook](../notebooks/acquire_traces.ipynb) is available for acquiring traces from the target devices.

You can find a code snnipet for the target hardware implementation at the end of this document. -->

# SAKURA-X/SASEBO-GIII sample design
If the SAKURA-X board is configured with a sample AES design distributed by AIST [here](https://www.risec.aist.go.jp/project/sasebo/),
your can use the target class [`SakuraX`](../lib/cw_plugins/targets/SakuraX.py) to control the target hardware.

## Target specific options
When connecting to the target hardware with `cw.target` routine,
you need to specify the serial port connected to USB interface of the SAKURA-X as `serial_port` keyword argument.
```python
import chipwhisperer as cw
from cw_plugins.targets import SakuraX
scope = cw.scope() # or Visa oscilloscope
target = cw.target(scope, SakuraX, serial_port="/dev/ttyUSB0") # modify the serial port as needed
```
Baud rate is defaulted to 115200, but you can change it by setting `baud` keyword argument.

# [ESP32_AES128](../hardware/ESP32_AES128/)

This is PlatformIO project for ESP32 with AES-128 hardware core.

Please change macro in `main.cpp` as you like.
* CPU_FREQ: CPU frequency in MHz
* LED_PIN: GPIO pin number for LED
* TRIGGER_PIN: GPIO pin number for trigger signal

If encrpition key is set, the LED is turned on.

The trigger signal is asserted when the encryption starts and deasserted when the encryption ends.

To communicate with the ESP32, the UART-USB converter is required.

## Target specific options
When connecting to target hardware with `con` method,
you need to specify the serial port connected to USB interface of the SAKURA-X as `serial_port` keyword argument.
```python
import chipwhisperer as cw
from cw_plugins.targets import ESP32
scope = cw.scope() # or Visa oscilloscope
targrt = cw.target(scope, ESP32, serial_port="/dev/ttyUSB0") # modify the serial port as needed
```

Baud rate is defaulted to 115200, but you can change it by setting `baud` keyword argument.

# SAKURA-X Shell
This is a our template of SAKURA-X board for easy evaluation of your cryptographic module.
For more information, please refer to its [repo](https://github.com/hal-lab-u-tokyo/sakura-x-shell).

This repository provides a driver template.

## [SakuraXShellBase](../lib/cw_plugins/targets/SakuraXShell.py)
This is an abstract class for Sakura-X Shell derived from `TargetTemplate` in Chipwhisperer.
If the following methods are implemented, the target can be used with `Chipwhisperer.capture_trace`.

* `getControl(self, **kwargs) -> SakuraXShellControlBase`
 return contoller object for target hardware on Sakura-X board (see below).
* `getExpected(self) -> bytes`: return the expected ciphertext.
* `loadEncryptionKey(self, key : bytes)`: set the encryption key.
* `loadInput(self, inputtext: bytes)`: load the plaintext.

## [SakuraXShellControlBase](../lib/cw_plugins/targets/SakuraXShell.py)
This is also an abstract class.
An instance of a derived class is an object to define the control of the target hardware on Sakura-X Shell, including key setting, sending plaintext, receiving ciphertext, and starting encryption.
The base class already defines communication methods with shell part of Sakura-X board.
Thus, the derived class at least needs to implement the following methods.

* constructor: `__init__(self, ser : serial.Serial, **kwargs)`
The constructor of the derived class requires at least one argument, `ser`, which is a serial object for communication with the shell part of Sakura-X board and call super constructor with `ser`.
When using `cw.target` routine, additional keyword arguments can be passed to the constructor.
In this way, you can handle the target specific options in the derived class.

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

In addition to the above abstract methods, the derived class may need to implement `isDone(self) -> bool` method to ensure the encryption is completed.

## Utility functions to get memory map on the target hardware
Vivado generates a hardware handoff file (`.hwh`) when generating the bitstream.
This file contains the address map of the hardware design.
To avoid hardcoding the address map in the driver, you can use the utility functions as follows:
```python
from cw_plugins.targets.utils import vivado_parse_memmap, ParseError
hwh_file = "path/to/hwh_file"
try:
	memmap = vivado_parse_memmap(hwh_file, "/controller_AXI_0")
except ParseError as e:
	# handle the error
```
`controller_AXI_0` refers to the instance name of the AXI master which translates the memory access requests from the control host PC to the target hardware.
As far as you use the default script pvovided in the sakura-x-shell repository, you don't need to change this instance name.
If there is no parsing error, it returns a `MemoryMap` instance.
For example, in your design, a cryptographic core named `crypto_core` is connected to the AXI master, you can get the address of the core as follows:
```python
base = memmap.crypto_core.base # base address of the crypto core
high = memmap.crypto_core.high # end address of the crypto core
```

## Example implementations of Sakura-X Shell
This repository provides example driver implementations for designs integrating AES-128 hardware cores in [sca_design_repo](https://github.com/hal-lab-u-tokyo/sca_design_repo) with the shell template.

Before using those, please configure the FPGAs on the board with the provided bitstreams or your built bitstreams.

### 1. RTL implementations of AES-128
To acquire traces from the RTL implementation, please use [`SakuraXShellExampleAES128BitRTL`](../lib/cw_plugins/targets/SakuraShellAESExamples.py) class.

The pin 1 of CN8 is used for the trigger signal, similar to SASEBO-GIII sample.

There are three versions of the RTL implementation, one is AIST implementation, Google ProjectVault implementation, and RSM masking implementation.

To specify the implementation, please set `implmentation="aist"`, `implmentation="google"` or `implmentation="rsm"` keyword argument to the `con` method or `cw.target` routine.
The serial port connected to USB interface of the SAKURA-X board must be set to `serial_port` keyword argument.

```python
import chipwhisperer as cw
from cw_plugins.targets import SakuraXShellExampleAES128BitRTL
scope = cw.scope()  # or Visa oscilloscope
target = cw.target(scope, SakuraXShellExampleAES128BitRTL,  serial_port="/dev/ttyUSB0", implementation="aist")
```

The library loads a default hardware handoff file included in the repository.
If you want to use your own hardware handoff file, please set `hwh_file` keyword argument to the path of the hardware handoff file.


### 2. HLS implementation of AES-128
To acquire traces from the HLS implementation, please use [`SakuraXShellExampleAES128BitHLS`](../lib/cw_plugins/targets/SakuraShellAESExamples.py) class.

The pin 1 of CN8 is used for the trigger signal, similar to SASEBO-GIII sample.

There are two versions of the HLS implementation, one is naive and the other is protected with RSM masking.
In similar way to the RTL implementation, please set `implmentation="naive"` or `implmentation="rsm"` keyword argument.

```python
import chipwhisperer as cw
from cw_plugins.targets import SakuraXShellExampleAES128BitHLS
scope = cw.scope()  # or Visa oscilloscope
target = cw.target(scope, SakuraXShellExampleAES128BitHLS,  serial_port="/dev/ttyUSB0", implementation="naive")
```

It also loads a default hardware handoff file, which is generated by Vivado to parse address map of the hardware design,
and you can specify your own hardware handoff file by setting `hwh_file` keyword argument to the path of the hardware handoff file.

# CW305-Shell
Similar to the SAKURA-X Shell, the CW305 Shell is a template for the CW305 board.
The original CW305 class and its data communication have some limitations.
A design based on our shell template and CW305Shell class provides ACK-based communication and more flexible data communication,
allowing vairable latency data response from the FPGA, data bus clock domain crossing, and so on.
For more information, please refer to its [repo](https://github.com/hal-lab-u-tokyo/cw305-shell).

This repository provides a driver template.

## [CW305ShellBase](../lib/cw_plugins/targets/CW305Shell.py)

Similar to `SakuraXShellBase`, this is an abstract class for CW305 Shell derived from `CW305` in Chipwhisperer.
A derived class needs to implement the following methods.

* `go(self)`: start the encryption.
* `getExpected(self) -> bytes`: return the expected ciphertext.
* `loadEncryptionKey(self, key: bytes)`: set the encryption key.
* `loadInput(self, inputtext: bytes)`: load the plaintext.
* `readOutput(self) -> bytes`: return the processed ciphertext.

In addition to the above abstract methods, the derived class may need to implement `isDone(self) -> bool` method to ensure the encryption is completed.

The utiliy functions to get memory map on the target hardware are also available in the same way as `SakuraXShellBase`.

### Overriding `_con` Method for Target Specific Options

When using `cw.target` routine to pass target-specific options, you need to override the `_con` method in the derived class implementation. Ensure that the base class `_con` method is called using `super()._con(scope, **kwargs)` to maintain proper initialization and functionality.

## Examples implementations of CW305 Shell
Similar to the SAKURA-X Shell, this repository also provides example drivers for the same designs on the CW305 board.

### 1. RTL implementation of AES-128
To acquire traces from the RTL implementation, please use  [`CW305ShellExampleAES128BitRTL`](../lib/cw_plugins/targets/CW305ShellAESExamples.py).

TIO4 pin is used for the trigger signal.
If you connect ChipWhisperer Lite as capture device through the ChipWhisperer 20-pin Connector, you just need to set the trigger signal to TIO4 pin as follows:
```python
scope = cw.scope()
scope.adc.basic_mode = "rising_edge"
scope.trigger.triggers = "tio4"
```
If you are using a general-purpose oscilloscope as the capture device, ensure that you probe the TIO4 pin located on the GPIO header (pin 33) of the CW305 board.
The three implementations of AES-128 encryption are also available for the CW305 Shell, similar to the SAKURA-X Shell. You can specify the implementation using the same `implementation` keyword argument (`"aist"`, `"google"`, or `"rsm"`). Additionally, the CW305 board allows configuration with a bitstream file directly in software. To specify the bitstream file, use the `bsfile` keyword argument when initializing the target class.


```python
import chipwhisperer as cw
from cw_plugins.targets import CW305ShellExampleAES128BitRTL
scope = cw.scope()  # or Visa oscilloscope
... # set trigger signal as described above
target = cw.target(scope, CW305ShellExampleAES128BitRTL, implementation="aist", bsfile="path/to/bitstream_file")
```
The library loads a default hardware handoff file included in the repository. If you want to use your own hardware handoff file, specify it using the `hwh_file` keyword argument.


### 2. HLS implementation of AES-128
To acquire traces from the HLS implementation, please use [`CW305ShellExampleAES128BitHLS`](../lib/cw_plugins/targets/CW305ShellAESExamples.py).

The TIO4 pin is also used for the trigger signal, and the same setup as the RTL implementation is required.
Similar to SAKURA-X, there are two versions of the HLS implementation, one is naive and the other is protected with RSM masking.


```python
import chipwhisperer as cw
from cw_plugins.targets import CW305ShellExampleAES128BitHLS
scope = cw.scope()  # or Visa oscilloscope
... # set trigger signal as described above
target = cw.target(scope, CW305ShellExampleAES128BitHLS, implementation="naive", bsfile="path/to/bitstream_file")
```


# VexRiscv_SCA
VexRiscv is a 32-bit RISC-V CPU core, and [VexRiscv_SCA](https://github.com/hal-lab-u-tokyo/VexRiscv_SakuraX) provides an implementations of VexRiscv for both SAKURA-X and CW305 boards.
For more information including software development kit, test runner scripts, and so on, please refer to its repo.

VexRiscv_SCA is also designed based on the Sakura-X Shell, and [`SakuraXVexRISCVControlBase`](../lib//cw_plugins/targets/SakuraXVexRISCV.py) defines additional methods for controlling the CPU core, derived from `SakuraXShellControlBase`.
As described above, the derived class still needs to implement the four methods `send_key`, `send_plaintext`, `run`, and `read_ciphertext`.

Its constructor requires the following arguments:
* ser (serial.Serial): serial object for communication with SAKURA-X board
* program (str): path to the elf binary file to be loaded to the CPU core

After instantiation, the CPU core boots with the program and starts running.

## Serial like communication with CPU core
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

For CW305 board, the driver class is [`CW305VexRISCVBase`](../lib/cw_plugins/targets/CW305VexRISCV.py).
It has almost the same methods as `SakuraXVexRISCVControlBase` except for the `_con` method arguments.
A path to program binary file must be set to the `program` keyword argument for `cw.target` routine.


## AES example

This repository includes software implementations of AES-128 encryption for VexRiscv.
[`SakuraXVexRISCVAESExample`](../lib/cw_plugins/targets/SakuraXVexRISCVAESExample.py) is a derived class of `SakuraXShellBase`.
There are two implementations of AES-128 encryption in the repository, one is unmasked and the other is masked.
`SakuraXVexRISCVAESExample` uses the unmasked implementation as default.
To use the masked implementation, please set keyword argument `masked` to `True` for `con` method.
Source code of each implementation can be found in `lib/cw_plugins/targets/aes_soft` directory.
If you prepare the SDK for VexRiscv_SDK, you can build the AES-128 program binary file by running `make` command in the directory.

### Target specific options for SAKURA-X

Similar to the shell-based examples, the pin 1 of CN8 is used for the trigger signal
and `serial_port` keyword argument is also required.

```python
import chipwhisperer as cw
from cw_plugins.targets import SakuraXVexRISCVAESExample
scope = cw.scope()  # or Visa oscilloscope
target = cw.target(scope, SakuraXVexRISCVAESExample, serial_port="/dev/ttyUSB0", masked=True)
```

### Target specific options for CW305

For CW305 board, `CW305RISCVAES128bit` is avaialble as follows:
```python
import chipwhisperer as cw
from cw_plugins.targets import CW305RISCVAES128bit
scope = cw.scope()  # or Visa oscilloscope
target = cw.target(scope, CW305RISCVAES128bit, bsfile="path/to/bitstream_file", masked=True)
```

For both SAKURA-X and CW305, the program binary file contained in this repository is loaded,
but you can specify your own program binary file by setting `program` keyword argument to the path of the binary file.


## Code snippet for just running AES-128 encryption (without trace acquisition)


```python
import chipwhisperer as cw

... # target instantiation here as described above

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
