###
#   Copyright (C) 2026 University of Tsukuba
#   
#   File:          /lib/cw_plugins/targets/SakuraXShell.py
#   Project:       chipwhisperer-enhanced-plugins
#   Author:        Takuya Kojima in University of Tsukuba (tkojima@lila.cs.tsukuba.ac.jp)
#   Created Date:  27-03-2024 18:15:49
#   Last Modified: 09-11-2026 21:07:07
###


from chipwhisperer.capture.targets._base import TargetTemplate
import serial
from pathlib import Path
from os import PathLike
import numpy as np
from abc import ABCMeta, abstractmethod
from .utils import vivado_parse_memmap,MemoryMap

from collections.abc import Iterable
import time
import warnings
import platform

PREAMBLE = 0x8
POSTAMBLE = 0x1
CMD_READ = 0x1
CMD_WRITE = 0x2
CMD_OK = 0x0
CMD_ERROR = 0x1
UNKNOWN_CMD = 0x2


class SakuraXShellBase(TargetTemplate, metaclass=ABCMeta):
    """Base Class for Sakura-X Shell Target

    """

    DEVICE_DIRECTORY = Path('/dev/sakura-x-shell')
    SYS_TTY_DIRECTORY = Path('/sys/class/tty')

    def __init__(self) -> None:
        """Initialize target state; connection options are supplied to con()."""
        super().__init__()
        self.connectStatus = False
        self.scope = None
        self.last_key = bytes()
        self.key = bytes()
        self.ser = None
        self.reset_ser = None
        self.serial_number = None
        self._memmap = None

    @property
    def memmap(self):
        return self._memmap

    @memmap.setter
    def memmap(self, value: MemoryMap | tuple[str | PathLike, str]):
        if isinstance(value, MemoryMap):
            self._memmap = value
        elif isinstance(value, tuple) and len(value) == 2:
            hwh_file, module_name = value
            self._memmap = vivado_parse_memmap(hwh_file, module_name)
        else:
            raise ValueError("memmap must be a MemoryMap instance or a tuple of (hwh_file, module_name)")

    ##############################################
    ############ Serial Port Handling ############
    ##############################################
    @staticmethod
    def _usb_interface(tty):
        """Find the USB interface ancestor of a Linux sysfs tty entry."""

        device = (tty / 'device').resolve()
        return next((p for p in (device, *device.parents)
                     if (p / 'bInterfaceNumber').is_file()), None)

    @classmethod
    def _find_usb_peer(cls, port, channel, serial_number):
        """Find the other FT2232H channel under the same physical USB device."""

        tty = cls.SYS_TTY_DIRECTORY / Path(port).resolve().name
    
        interface = cls._usb_interface(tty)
        if interface is None:
            raise RuntimeError(f'Cannot identify USB interface for {port}; specify both ports')
    
        usb = interface.parent
        if ((usb / 'idVendor').read_text().strip().lower() != '0403'
                or (usb / 'idProduct').read_text().strip().lower() != '6010'):
            raise RuntimeError(f'{port} is not an FT2232H port (0403:6010)')
        if (interface / 'bInterfaceNumber').read_text().strip() != channel:
            raise RuntimeError(f'{port} must be USB interface {channel}')
        
        serial_file = usb / 'serial'
        detected_serial = serial_file.read_text().strip() if serial_file.exists() else None
        if serial_number is not None and serial_number != detected_serial:
            raise RuntimeError(f'{port} does not match serial_number={serial_number!r}')

        peer_channel = '01' if channel == '00' else '00'
        peers = []
        for candidate in cls.SYS_TTY_DIRECTORY.glob('ttyUSB*'):
            peer = cls._usb_interface(candidate)
            if (peer is not None and peer.parent == usb
                    and (peer / 'bInterfaceNumber').read_text().strip() == peer_channel):
                peers.append(str(Path('/dev') / candidate.name))

        if len(peers) != 1:
            raise RuntimeError(f'Expected one USB interface {peer_channel} tty paired with '
                               f'{port}, found {len(peers)}; specify both ports')

        return detected_serial, peers[0]

    @classmethod
    def _resolve_ports(cls, serial_number, data_port, reset_port):
        """Resolve a pair of ports without opening either channel."""

        # If both ports are specified, just return them.
        if data_port is not None and reset_port is not None:
            return serial_number, str(data_port), str(reset_port)

        system = platform.system()
        if system != 'Linux':
            system = 'macOS' if system == 'Darwin' else system
            missing = ', '.join(name for name, value in
                                (('data_port', data_port), ('reset_port', reset_port))
                                if value is None)
            raise RuntimeError(
                f'SAKURA-X Shell automatic port discovery is supported only on Linux, '
                f'not {system}. Specify both data_port (Channel A) and reset_port '
                f'(Channel B). Missing: {missing}. serial_number alone cannot '
                'select ports on this OS.')

        # One explicit port identifies the physical USB device directly.
        if data_port is not None:
            selected, peer = cls._find_usb_peer(data_port, '00', serial_number)
            return selected, str(data_port), peer
        if reset_port is not None:
            selected, peer = cls._find_usb_peer(reset_port, '01', serial_number)
            return selected, peer, str(reset_port)

        # enumerate all candidate boards based on device directory
        root = cls.DEVICE_DIRECTORY

        # In the case of a manually supplied serial number, only consider the matching board.
        candidates = [root / serial_number] if serial_number is not None else (
            sorted(root.iterdir()) if root.is_dir() else [])

        # Only consider candidates that have both data and reset channels.
        candidates = [p for p in candidates
                      if p.is_dir() and (p / 'data').exists() and (p / 'reset').exists()]

        # No candidates found    
        if not candidates:
            raise RuntimeError(
                f'No matching SAKURA-X Shell data/reset pair in {root}'
                f' (serial_number={serial_number!r}). Check the udev rules and USB connection, '
                'or specify both data_port and reset_port.')

        # Multiple candidates found but no serial number specified; cannot choose automatically.
        if len(candidates) > 1:
            raise RuntimeError('Multiple SAKURA-X Shell boards found: '
                               + ', '.join(p.name for p in candidates)
                               + '. Specify serial_number.')
        
        board = candidates[0]

        return board.name, str(board / 'data'), str(board / 'reset')

    def _con(self, scope, serial_number = None, data_port = None, reset_port = None, serial_port = None, **kwargs):
        """
        Auto detection and serial number selection work only on Linux with udev rules installed.
        For macOS and Windows, specify both data_port and reset_port. 
        With multiple boards, if you do not specify serial_number, the first board found will be used.
        """

         # check if hwh file exists in the same directory as bitstream
        if "hwh_file" in kwargs and kwargs["hwh_file"] is not None:
            print(kwargs["hwh_file"])
            hwh_path = Path(kwargs["hwh_file"])
            if hwh_path.exists():
                self.memmap = str(hwh_path), "/controller_AXI_0"

        if serial_port is not None:
            if data_port is not None:
                raise TypeError('Specify only one of data_port and deprecated serial_port')
            warnings.warn('serial_port is deprecated; use data_port instead',
                          DeprecationWarning, stacklevel=2)
            data_port = serial_port
        if self.ser is not None or self.reset_ser is not None:
            raise RuntimeError('Already connected; disconnect before reconnecting')
    
        selected, data_port, reset_port = self._resolve_ports(serial_number, data_port, reset_port)
    
        if (Path(data_port).resolve() == Path(reset_port).resolve()
                or data_port.casefold() == reset_port.casefold()):
            raise ValueError('data_port and reset_port must be different ports')
        try:
            # Set RTS inactive before open; True asserts the active-low RTS# pin.
            self.reset_ser = serial.Serial(port=None, timeout=1,
                                           rtscts=False, dsrdtr=False)
            self.reset_ser.rts = False
            self.reset_ser.dtr = False
            self.reset_ser.port = reset_port
            self.reset_ser.open()
            # Allow recovery if the OS/driver briefly asserted RTS during open.
            time.sleep(1)
            self.ser = serial.Serial(data_port, timeout=1, write_timeout=1)
            self.scope = scope
            self.serial_number = selected
            self.last_key = bytes()
            self.reset_interface()
            self.soft_reset()
        except Exception:
            self._close_ports()
            raise


    def reset_interface(self):
        """Reset both FPGA communication paths and reinitialize the controller.

        Discards pending data trasnfers and reinitializes the controller's state.
        """
        if (self.ser is None or self.reset_ser is None
                or not self.ser.is_open or not self.reset_ser.is_open):
            raise RuntimeError('SAKURA-X Shell is not connected')

        self.last_key = bytes()
        self.key = bytes()
        try:
            self.reset_ser.rts = True
            try:
                # Do not flush(): a stalled transfer must be discarded, not awaited.
                self.ser.reset_output_buffer()
                time.sleep(0.1)
                self.ser.reset_input_buffer()
            finally:
                self.reset_ser.rts = False
            time.sleep(1)
            self.ser.reset_input_buffer()
        except Exception:
            self._close_ports()
            raise

    def _dis(self):
        self._close_ports()

    def _close_ports(self):
        """Release both channels even if controller shutdown fails."""
        data, reset = self.ser, self.reset_ser
        self.ser = self.reset_ser = self.ctrl = self.scope = None
        self.serial_number = None
        self._control_kwargs = {}
        self.last_key = bytes()
        try:
            if data is not None:
                data.close()
        finally:
            if reset is not None:
                try:
                    if reset.is_open:
                        reset.rts = False
                finally:
                    reset.close()

    def reset(self):
        self.reset_interface()
        self.soft_reset()
        self.last_key = bytes()


    ###########################################################
    ############# FPGA Communication Functionality ############
    ###########################################################

    # Command format
    # +------------------+--------------+--------------------+---------+-------------------+
    # |      4 bits      |    4 bits    |       4 bits       | 32 bits |      4 bits       |
    # +------------------+--------------+--------------------+---------+-------------------+
    # | Preamble 2'b1000 | Command type | Command Attributes |   Args  | Postamble 2'b0001 |
    # +------------------+--------------+--------------------+---------+-------------------+
    # Command type
    # * 0x0: Reset
    # * 0x1: Read data
    # * 0x2: Write data
    # Read data commnad
    # * Attributes: data length (0 means 1 word, 15 means 16 words)
    # * Args: address
    # Write data command
    # Same as read data command
    # Reset command
    # * arributes must be 0
    # * Args are not used

    def __wait_for_response(self):
        resp = self.ser.read(1)
        if len(resp) == 0:
            raise RuntimeError("""Timeout Error. Possiblly controller is stuck.
            Check the status LEDs. If the controller is stuck, LED 16 will be turned on.
            When the input fifo is full and the controller cannot accpet further input, LED 14 will be turned off. In this case, you need to reset the controller by calling hard_reset() method of the target.""")
        elif resp[0] == CMD_OK:
            return
        elif resp[0] == CMD_ERROR:
            raise RuntimeError("Command Error")
        elif resp[0] == UNKNOWN_CMD:
            raise RuntimeError("Unknown Command")
        else:
            raise RuntimeError("Unexpected Response received")

    def fpga_write(self, addr : int, data : Iterable[int] | int):
        # check data type
        if not hasattr(data, "__iter__"):
            data = [data]
        else:
            data = list(data)

        word_len = len(data)
        if not (1 <= word_len <= 16):
            raise ValueError("Data length must be between 1 and 16")
        data_bin = b''.join([int(d).to_bytes(4, 'big') for d in data])
        cmd = f"{PREAMBLE:1X}_{CMD_WRITE:1X}_{word_len-1:1X}_{addr:08X}_{POSTAMBLE:1X}"
        cmd_bin = int(cmd,16).to_bytes(6, 'big')
        self.ser.write(cmd_bin)
        self.__wait_for_response()
        self.ser.write(data_bin)

    def fpga_read(self, addr : int, data_len : int):
        """
            Read data from Sakura-X-Shell Controller
            Args:
                addr (int): Start address
                data_len (int): Data length in words
            Returns:
                numpy array (data type = numpy.uint32)

        """
        if not (1 <= data_len <= 16):
            raise ValueError("Data length must be between 1 and 16")
        cmd = f"{PREAMBLE:1X}_{CMD_READ:1X}_{data_len-1:1X}_{addr:08X}_{POSTAMBLE:1X}"
        cmd_bin = int(cmd,16).to_bytes(6, 'big')
        self.ser.write(cmd_bin)
        self.__wait_for_response()
        read_bin = self.ser.read(data_len * 4)
        if len(read_bin) != data_len * 4:
            raise RuntimeError(
                f"FPGA read returned {len(read_bin)} bytes; expected {data_len * 4} bytes")

        return np.frombuffer(read_bin, dtype=">u4").astype(np.uint32)

    def soft_reset(self):
        """Send reset signal to modules on the Kintex-7 FPGA
        """
        cmd = f"{PREAMBLE:1X}_00_{0x0:08X}_{POSTAMBLE:1X}"
        cmd_bin = int(cmd,16).to_bytes(6, 'big')
        self.ser.write(cmd_bin)
        self.__wait_for_response()
        # wait encryption module to be ready
        time.sleep(1)

    
    def flush(self):
        self.ser.flush()
        self.ser.reset_output_buffer()

    def fpga_led_on(self, led):
        """Turn on onbaord LEDs

            Args:
                led (int): LED number to turn on (0 to 9)

            Note:
                This function does not change the state of other LEDs.
                This function needs hardware handoff file (.hwh) to be loaded.
        """

        if self.memmap is None:
            raise RuntimeError("No memory map is available. Load .hwh when connecting to the target")

        try:
            current_leds = self.fpga_read(self.memmap.axi_usr_dip_led.base + 8, 1) # see AMD DS744 datasheet
        except AttributeError:
            raise RuntimeError("AXI GPIO module for LED control is not found in the configured FPGA design. Please check the .hwh file and the design.")

        if led > 9 or led < 0:
            raise ValueError("LED number must be between 0 and 9")

        next_leds = int(current_leds[0]) | (1 << led)
        self.fpga_write(self.memmap.axi_usr_dip_led.base + 8, next_leds)

    def fpga_led_off(self, led):
        """Turn off onbaord LEDs

            The arguments are the same as fpga_led_on.
        """
        if self.memmap is None:
            raise RuntimeError("No memory map is not available. Load .hwh when connecting to the target")

        try:
            current_leds = self.fpga_read(self.memmap.axi_usr_dip_led.base + 8, 1)
        except AttributeError:
            raise RuntimeError("AXI GPIO module for LED control is not found in the configured FPGA design. Please check the .hwh file and the design.")

        if led > 9 or led < 0:
            raise ValueError("LED number must be between 0 and 9")

        next_leds = int(current_leds[0]) & ~(1 << led)

        self.fpga_write(self.memmap.axi_usr_dip_led.base + 8, next_leds)


    ###########################################################
    #############         abstract methods        #############
    ###########################################################
    @abstractmethod
    def go(self):
        pass

    @abstractmethod
    def getExpected(self):
        """Return expected ciphertext.
            If readed ciphertext is not equal to this value, the encryption is regarded as failed.
        """
        pass

    @abstractmethod
    def loadEncryptionKey(self, key):
        """Load encryption key to the target module"""
        pass

    @abstractmethod
    def loadInput(self, inputtext):
        """Load input text to the target module"""
        pass

    @abstractmethod
    def readOutput(self):
        """Read ciphertext from the target module"""
        pass


    def getName(self):
        return "Base Class for Sakura-X Shell"

    def isDone(self):
        """Check if the target module has finished the encryption"""
        return True


    # Wrapper methods for compatibility with ChipWhisperer.capture_trace
    def set_key(self, key, **kwargs):
        """Set encryption key"""
        self.key = key
        if self.last_key != key:
            self.loadEncryptionKey(key)


    def simpleserial_read(self, cmd, pay_len, **kwargs):
        """Read data from target"""
        if cmd == "r":
            return self.readOutput()
        else:
            raise ValueError("Unknown command {}".format(cmd))

    def simpleserial_write(self, cmd, data, end=None):
        if cmd == 'p':
            self.loadInput(data)
            self.go()
        elif cmd == 'k':
            self.loadEncryptionKey(data)
        else:
            raise ValueError("Unknown command {}".format(cmd))

    def is_done(self):
        return self.isDone()


