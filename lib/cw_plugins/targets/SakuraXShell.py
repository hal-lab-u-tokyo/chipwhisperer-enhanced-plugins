###
#   Copyright (C) 2024 The University of Tokyo
#   
#   File:          /lib/cw_plugins/targets/SakuraXShell.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  27-03-2024 18:15:49
#   Last Modified: 31-03-2024 02:51:19
###

from chipwhisperer.capture.targets._base import TargetTemplate
import serial
from serial.serialutil import SerialException
import serial.tools.list_ports
import numpy as np
from Crypto.Cipher import AES
from collections.abc import Iterable, Callable
import time

PREAMBLE = 0x8
POSTAMBLE = 0x1
CMD_READ = 0x1
CMD_WRITE = 0x2
CMD_OK = 0x0
CMD_ERROR = 0x1
UNKNOWN_CMD = 0x2

class SakuraXShellControlBase:
    """Base Interface for Sakura-X Shell Controller"""
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
        cmd = f"{PREAMBLE:1X}_{CMD_READ:1X}_{length-1:1X}_{addr:08X}_{POSTAMBLE:1X}"
        cmd_bin = int(cmd,16).to_bytes(6, 'big')
        self.ser.write(cmd_bin)
        self.__wait_for_response()
        read_bin = self.ser.read(length * 4)

        return [int.from_bytes(read_bin[4*i:4*i+4], byteorder='big') for i in range(length)]


    def reset_command(self):
        print("Reset Command")
        cmd = f"{PREAMBLE:1X}_00_{0x0:08X}_{POSTAMBLE:1X}"
        cmd_bin = int(cmd,16).to_bytes(6, 'big')
        self.ser.write(cmd_bin)
        self.__wait_for_response()

    def flush(self):
        self.ser.flush()
        self.ser.reset_output_buffer()

    def reset(self):
        self.flush()
        self.reset_command()

    def close(self):
        self.ser.close()

    def send_key(self, key : bytes):
        pass

    def send_plaintext(self, plaintext : bytes):
        pass

    def run(self):
        pass
    def read_ciphertext(self, byte_len : int = 8):
        pass

    def isDone(self):
        return True

class SakuraXShellBase(TargetTemplate):

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
        self.cipher = None
        self.last_key = [0 for _ in range(16)]

    def getControl(self, **kwargs) -> SakuraXShellControlBase:
        pass

    def getName(self):
        return "Base Class for Sakura-X Shell"

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
        return np.frombuffer(self.ctrl.read_ciphertext(len(self.last_key)),\
                            dtype=np.uint8)

    def go(self):
        self.ctrl.run()

    def getExpected(self):
        ct = self.cipher.encrypt(bytes(self.input))
        ct = bytearray(ct)
        return ct

    def loadEncryptionKey(self, key):
        self.ctrl.send_key(bytes(key))
        self.last_key = key
        self.cipher = AES.new(bytes(key), AES.MODE_ECB)

    def loadInput(self, inputtext):
        self.input = inputtext
        self.ctrl.send_plaintext(bytes(inputtext))

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

# Example Implementations for Sakura-X Shell
class SakuraXShellExampleAES128BitRTLControl(SakuraXShellControlBase):
    ADDRESS_MAP = {
        "key": 0x0,
        "plaintext": 0x40,
        "ciphertext": 0x80,
        "control": 0x30,
    }
    KEY_READY_BIT = 0x2
    PT_READY_BIT = 0x1
    def __init__(self, ser, address_base = 0x44A0_0000, **kwargs):
        super().__init__(ser)
        self.address_base = address_base

    def send_key(self, key : bytes):
        key_words = [int.from_bytes(key[4*i:4*i+4], byteorder='big') for i in range(4)]
        self.write_data(self.address_base + self.ADDRESS_MAP["key"], key_words[::-1])

    def send_plaintext(self, plaintext : bytes):
        pt_words = [int.from_bytes(plaintext[4*i:4*i+4], byteorder='big') for i in range(4)]
        self.write_data(self.address_base + self.ADDRESS_MAP["plaintext"], pt_words[::-1])

    def run(self):
        self.write_data(self.address_base + self.ADDRESS_MAP["control"], \
                         [self.KEY_READY_BIT | self.PT_READY_BIT])

    def read_ciphertext(self, byte_len : int = 8):
        read_words = self.read_data(self.address_base + self.ADDRESS_MAP["ciphertext"], byte_len // 4)
        ct = b''
        for w in read_words[::-1]:
            ct += w.to_bytes(4, byteorder='big')
        return ct


class SakuraXShellExampleAES128BitRTL(SakuraXShellBase):
    def getControl(self, **kwargs) -> SakuraXShellControlBase:
        return SakuraXShellExampleAES128BitRTLControl(self.ser, **kwargs)

class SakuraXShellExampleAES128BitHLSControl(SakuraXShellControlBase):
    CTRL_ADDRESS_MAP = {
        "control": 0x0,
        "key_offset": 0x10,
        "plaintext_offset": 0x18,
        "ciphertext_offset": 0x20,
    }
    CTRL_AP_START = 0x1
    CTRL_AP_DONE = 0x2
    CTRL_AP_IDLE = 0x4
    CTRL_AP_READY = 0x8
    CTRL_AP_CONTINUE = 0x10

    def __init__(self, ser, control_address = 0x0, bram_address = 0xC000_0000, **kwargs):
        super().__init__(ser)
        self.reset_command()
        self.control_address = control_address
        self.key_address = bram_address
        self.plaintext_address = bram_address + 0x10
        self.ciphertext_address = bram_address + 0x20

        self.write_data(self.control_address + self.CTRL_ADDRESS_MAP["key_offset"], [self.key_address])
        self.write_data(self.control_address + self.CTRL_ADDRESS_MAP["plaintext_offset"], [self.plaintext_address])
        self.write_data(self.control_address + self.CTRL_ADDRESS_MAP["ciphertext_offset"], [self.ciphertext_address])

        self.write_data(self.control_address + self.CTRL_ADDRESS_MAP["control"], [self.CTRL_AP_CONTINUE])


    def get_status(self):
        stat = self.read_data(self.control_address + self.CTRL_ADDRESS_MAP["control"], 1)[0]
        ap_start = (stat & self.CTRL_AP_START) != 0
        ap_done = (stat & self.CTRL_AP_DONE) != 0
        ap_idle = (stat & self.CTRL_AP_IDLE) != 0
        ap_ready = (stat & self.CTRL_AP_READY) != 0
        ap_continue = (stat & self.CTRL_AP_CONTINUE) != 0
        return {"ap_start": ap_start, "ap_done": ap_done, "ap_idle": ap_idle, "ap_ready": ap_ready, "ap_continue": ap_continue}


    def send_key(self, key : bytes):
        data = []
        for i in range(4):
            data.append(int.from_bytes(key[4*i:4*i+4], byteorder='little'))
        self.write_data(self.key_address, data)

    def send_plaintext(self, plaintext : bytes):
        data = []
        for i in range(4):
            data.append(int.from_bytes(plaintext[4*i:4*i+4], byteorder='little'))
        self.write_data(self.plaintext_address, data)

    def run(self):
        stat = self.get_status()
        if stat["ap_idle"]:
            self.write_data(self.control_address + self.CTRL_ADDRESS_MAP["control"], [self.CTRL_AP_START])
        else:
            raise RuntimeError("HLS IP is not idle")

    def read_ciphertext(self, byte_len : int = 8):
        stat = self.get_status()
        if not stat["ap_done"]:
            raise RuntimeError("HLS IP is not done")

        read_words = self.read_data(self.ciphertext_address, byte_len // 4)
        ct = b''
        for w in read_words:
            ct += w.to_bytes(4, byteorder='little')
        self.write_data(self.control_address + self.CTRL_ADDRESS_MAP["control"], [self.CTRL_AP_CONTINUE])
        return ct

    def isDone(self):
        stat = self.get_status()
        return stat["ap_done"]

class SakuraXShellExampleAES128BitHLS(SakuraXShellBase):
    def getControl(self, **kwargs) -> SakuraXShellControlBase:
        return SakuraXShellExampleAES128BitHLSControl(self.ser, **kwargs)

