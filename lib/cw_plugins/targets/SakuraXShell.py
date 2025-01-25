###
#   Copyright (C) 2024 The University of Tokyo
#   
#   File:          /lib/cw_plugins/targets/SakuraXShell.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  27-03-2024 18:15:49
#   Last Modified: 25-01-2025 15:30:22
###

from chipwhisperer.capture.targets._base import TargetTemplate
import serial
from serial.serialutil import SerialException
import serial.tools.list_ports
import numpy as np
from abc import ABCMeta, abstractmethod

from collections.abc import Iterable
import time

PREAMBLE = 0x8
POSTAMBLE = 0x1
CMD_READ = 0x1
CMD_WRITE = 0x2
CMD_OK = 0x0
CMD_ERROR = 0x1
UNKNOWN_CMD = 0x2

class SakuraXShellControlBase(metaclass=ABCMeta):
    """Base Interface for Sakura-X Shell Controller
        This class is an abstract class for Sakura-X Shell Controller.
        Derived class must implement the following methods depending on the custom hardware on the Kintex-7 FPGA.
        * send_key: Send encryption key to the encryption module
        * send_plaintext: Send plaintext to the encryption module
        * run: Start encryption
        * read_ciphertext: Read ciphertext from the encryption module
    """
    ADDRESS_MAP = {
        "key": 0x0,
        "plaintext": 0x10,
        "ciphertext": 0x20,
    }
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

    def __init__(self, ser) -> None:
        self.ser = ser
        self.ser.reset_input_buffer()

    def flush(self):
        self.ser.flush()
        self.ser.reset_output_buffer()

    def __wait_for_response(self):
        resp = self.ser.read(1)
        if len(resp) == 0:
            raise RuntimeError("Timeout Error")
        elif resp[0] == CMD_OK:
            return
        elif resp[0] == CMD_ERROR:
            raise RuntimeError("Command Error")
        elif resp[0] == UNKNOWN_CMD:
            raise RuntimeError("Unknown Command")
        else:
            raise RuntimeError("Unexpected Response received")

    def write_data(self, addr : int, data : Iterable):
        word_len = len(data)
        if word_len > 16:
            raise ValueError("Data length must be less than or equal to 16")
        cmd = f"{PREAMBLE:1X}_{CMD_WRITE:1X}_{word_len-1:1X}_{addr:08X}_{POSTAMBLE:1X}"
        cmd_bin = int(cmd,16).to_bytes(6, 'big')
        self.ser.write(cmd_bin)
        self.__wait_for_response()
        data_bin = b''.join([d.to_bytes(4, 'big') for d in data])
        self.ser.write(data_bin)

    def read_data(self, addr : int, length : int):
        """
            Read data from Sakura-X-Shell Controller
            Args:
                addr (int): Start address
                length (int): Data length in words
            Returns:
                list of 32 bit integers

        """
        if not (1 <= length <= 16):
            raise ValueError("Data length must be between 1 and 16")
        cmd = f"{PREAMBLE:1X}_{CMD_READ:1X}_{length-1:1X}_{addr:08X}_{POSTAMBLE:1X}"
        cmd_bin = int(cmd,16).to_bytes(6, 'big')
        self.ser.write(cmd_bin)
        self.__wait_for_response()
        read_bin = self.ser.read(length * 4)

        return [int.from_bytes(read_bin[4*i:4*i+4], byteorder='big') for i in range(length)]


    def reset_command(self):
        """Send reset signal to modules on the Kintex-7 FPGA
        """
        cmd = f"{PREAMBLE:1X}_00_{0x0:08X}_{POSTAMBLE:1X}"
        cmd_bin = int(cmd,16).to_bytes(6, 'big')
        self.ser.write(cmd_bin)
        self.__wait_for_response()
        # wait encryption module to be ready
        time.sleep(1)


    def reset(self):
        self.flush()
        self.reset_command()

    def close(self):
        self.ser.close()

    @abstractmethod
    def send_key(self, key : bytes):
        pass

    @abstractmethod
    def send_plaintext(self, plaintext : bytes):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def read_ciphertext(self, byte_len : int = 8):
        pass

    def isDone(self):
        return True

class SakuraXShellBase(TargetTemplate, metaclass=ABCMeta):
    """Base Class for Sakura-X Shell Target

    """

    def __init__(self) -> None:
        """
            Constructor for Sakura-X Shell Target
            Args:
                ShellControl (Callable): function to instantiate SakuraXShellControlBase derived class

        """
        super().__init__()
        self.connectStatus = False
        self.ctrl = None
        self.scope = None
        self.last_key = bytes()
        self.key = bytes()

    def _con(self, scope, serial_port = None, baud = 115200, **kwargs):
        if serial_port is None:
            serial_port = "/dev/ttyUSB0"
        try:
            self.ser = serial.Serial(serial_port, baud, timeout=1)
        except SerialException as E:
            raise RuntimeError(E.args)
        self.scope = scope
        self.ctrl = self.getControl(**kwargs)

    def reset(self):
        self.ctrl.reset()

    def _dis(self):
        self.ctrl.close()
        self.ctrl = None
        self.scope = None

    def flush(self):
        self.ctrl.flush()

    def readOutput(self):
        return np.frombuffer(self.ctrl.read_ciphertext(self.textLen()),\
                            dtype=np.uint8)

    def go(self):
        self.ctrl.run()

    def getName(self):
        return "Base Class for Sakura-X Shell"

    # Abstract methods
    @abstractmethod
    def getControl(self, **kwargs) -> SakuraXShellControlBase:
        """Derived class must implement this method to instantiate SakuraXShellControlBase derived class"""
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

    def isDone(self):
        return self.ctrl.isDone()

