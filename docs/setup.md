
# Required Python Packages
```
git clone https://github.com/newaetech/chipwhisperer.git
cd chipwhisperer
pip3 install .
pip3 install -r jupyter/requirements.txt
```

# Install toolbox packages
```
# clone & cd to this repository
pip3 install .
```

# VISA Driver
To operate an oscilloscope controllable by VISA, you need the NI-VISA or PyVISA-py driver. Install either one that you prefer.

## NI-VISA
The following instructions are for version 2023Q04. When installing, please adjust accordingly for the latest version.

Download NI Device driver from https://www.ni.com/ja/search.html?pg=1&ps=10&sb=%2Brelevancy&sn=catnav:sup.dwl.ndr.

```
unzip  NILinux2023Q4DeviceDrivers.zip
```

### ubuntu 22.04
```
sudo apt install ./ni-ubuntu2204-drivers-2023Q4.deb
sudo apt update
sudo apt install ni-visa ni-visa-devel ni-hwcfg-utility
sudo dkms autoinstall
reboot
```

If instruments are not recoginzed correctly, `sudo rmmod usbtmc` perhaps solves the problem.

### RHEL,CentOS, Rocky 8
```
sudo dnf localinstall ni-rhel8-drivers-2023Q4.rpm
sudo dnf install ni-visa ni-visa-devel ni-hwcfg-utility
sudo dkms autoinstall
```

## PyVISA-py
```
pip3 install pyvisa-py
```

# Udev Rule files
Please copy the rules file from this repository or modify it to match the device you are using. 'lsusb' command is useful for checking the Vendor ID and Product ID.

```
# for SAKURA-X board
sudo cp udev-rules/99-sakura-x.rules /etc/udev/rules.d/
# for keysight MSOX4104A
sudo cp udev-rules/99-keysight-oscilloscope.rules /etc/udev/rules.d/
# for rigol MSO8104A
sudo cp udev-rules/99-rigol-oscilloscope.rules /etc/udev/rules.d/
```