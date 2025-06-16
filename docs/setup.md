# Supported Systems

This framework supports Linux and macOS. For Windows users, it is recommended to use either WSL (Windows Subsystem for Linux) or [Docker](./docker.md) to set up the environment.
For Linux users, please refer to the [Linux Prerequisites](#prerequisites-for-linux) section below for the necessary steps to set up the environment.
For macOS users, an installation script is provided to simplify the setup process.
Please refer to the [macOS Installation Guide](#installation-script-on-macos) section below for details.

# Prerequisites for Linux
This framework needs [Chipwhisperer](https://github.com/newaetech/chipwhisperer) as a core componet.
Please install it first acording to the official documentation.
Currently, we tested this framework with Chipwhisperer 5.7.0.
At least, the following commands are required to install Chipwhisperer and its Jupyter Notebook environment.

```
git clone https://github.com/newaetech/chipwhisperer.git -b 5.7.0
cd chipwhisperer
pip3 install .
pip3 install -r jupyter/requirements.txt
```

## Install plugin packages
```
git clone --recursive https://github.com/hal-lab-u-tokyo/chipwhisperer-enhanced-plugins
cd chipwhisperer-enhanced-plugins
pip3 install . -v
```
To build the C++ libraries, you need several additional packages, like `cmake` and c++ compilers.
Docker files in [docker](../docker) directory would be helpful to understand the required packages.
Cmake will automatically check the required packages to build each acceleration library.
For example, to build the CUDA acceleration library, Cmake have to identify the CUDA toolkit installed in your system.
Otherwise, the CUDA acceleration library will not be built.
If the libraries you want to use are not built, please see the CMake output during the installation process.

# Installation script on macOS
[install_mac.sh](../install_mac.sh) is provided at the root directory of this repository to simplify the installation process on macOS.

## Prerequisites for Using the macOS Installation Script

Before running the installation script, ensure the following prerequisites are met:

1. **Python Installation**: Python 3.10 or later is recommended. Verify that Python is installed on your system.
2. **Development Tools** (Optional): Install Xcode or a C++ compiler if you plan to build C++ libraries during the setup process.

### Virtual Environment Setup
The installation script will prompt you to create a virtual environment during execution. Follow the on-screen instructions to complete this step.

## Test the Installation of the C++ Libraries
This repository includes pytest-based tests to verify the installation of the C++ libraries.
Go to pytest directory and run the following command to execute the tests:
```
cd chipwhisperer-enhanced-plugins/pytest
pytest -v
```


# Steps to use VISA-compatible oscilloscopes
Users who do not need to use VISA-compatible oscilloscopes to capture traces can skip this section.
This instructions are mainly for Linux users, but similar steps may be applied to macOS users.

## Installing VISA Driver
To operate an oscilloscope controllable by VISA, you need the NI-VISA or PyVISA-py driver. Install either one that you prefer.

### Option 1: NI-VISA
The following instructions are for version 2023Q04 for Linux.
When installing, please adjust accordingly for your preferred version and operating system.

Download NI Device driver from https://www.ni.com/ja/search.html?pg=1&ps=10&sb=%2Brelevancy&sn=catnav:sup.dwl.ndr.

```
unzip  NILinux2023Q4DeviceDrivers.zip
```

For ubuntu 22.04 users, you can use the following commands to install the drivers:
```
sudo apt install ./ni-ubuntu2204-drivers-2023Q4.deb
sudo apt update
sudo apt install ni-visa ni-visa-devel ni-hwcfg-utility
sudo dkms autoinstall
reboot
```

If instruments are not recoginzed correctly, `sudo rmmod usbtmc` perhaps solves the problem.

For RHEL-8 or RHEL-based distributions (e.g., Rocky Linux, AlmaLinux), you can use the following commands to install the drivers:
```
sudo dnf localinstall ni-rhel8-drivers-2023Q4.rpm
sudo dnf install ni-visa ni-visa-devel ni-hwcfg-utility
sudo dkms autoinstall
```

### Option 2: PyVISA-py
This is a pure Python implementation of the VISA interface, which can be used as an alternative to NI-VISA.
```
pip3 install pyvisa-py
```

## Installing udev rules
Please copy the rules file from this repository or modify it to match the device you are using. 'lsusb' command is useful for checking the Vendor ID and Product ID.

```
# for SAKURA-X board
sudo cp udev-rules/99-sakura-x.rules /etc/udev/rules.d/
# for keysight MSOX4104A
sudo cp udev-rules/99-keysight-oscilloscope.rules /etc/udev/rules.d/
# for rigol MSO8104A
sudo cp udev-rules/99-rigol-oscilloscope.rules /etc/udev/rules.d/
```