###
#   Copyright (C) 2024 The University of Tokyo
#   
#   File:          /lib/cw_plugins/targets/ESP32.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  27-03-2024 18:16:05
#   Last Modified: 27-03-2024 18:16:09
###

from chipwhisperer.capture.targets._base import TargetTemplate
from Crypto.Cipher import AES
import numpy as np
import serial
from serial.serialutil import SerialException

class ESP32Control:
    CMD_SET_KEY 		= 0x11
    CMD_SET_PLAINTEXT	= 0x12
    CMD_ENCRYPT			= 0x13
    CMD_GET_CIPHERTEXT	= 0x14
    CMD_GET_DEBUG		= 0x15

    def __init__(self, ser) -> None:
        self.ser = ser
        self.ser.reset_input_buffer()

    def flush(self):
        self.ser.flush()
        self.ser.reset_output_buffer()

    def reset(self):
        self.flush()

    def close(self):
        self.ser.close()

    def send_key(self, key : bytes):
        buf = b""
        buf += self.CMD_SET_KEY.to_bytes(1, 'big')
        buf += key
        self.ser.write(buf)

    def send_plaintext(self, plaintext : bytes):
        buf = b""
        buf += self.CMD_SET_PLAINTEXT.to_bytes(1, 'big')
        buf += plaintext
        self.ser.write(buf)

    def run(self):
        buf = b""
        buf += self.CMD_ENCRYPT.to_bytes(1, 'big')
        self.ser.write(buf)
        stat = self.ser.read(1)
        if stat[0] != 0:
            print(stat)
            print(self.read_ciphertext(16))
            print()
            raise RuntimeError("Encryption failed")

    def read_ciphertext(self, byte_len : int = 8):
            buf = b""
            buf += self.CMD_GET_CIPHERTEXT.to_bytes(1, 'big')
            self.ser.write(buf)
            return self.ser.read(16)

class ESP32(TargetTemplate):
    _name = "ESP32"

    def __init__(self):
        super().__init__()
        self.ctrl = None
        self.scope = None
        self.cipher = None
        self.last_key = [0 for _ in range(16)]

    def getName(self):
        return self

    def _con(self, scope, serial_port = None, baud = 115200):
        if serial_port is None:
            serial_port = "/dev/ttyUSB0"
        try:
            self.ser = serial.Serial(serial_port, baud, timeout=1)
        except SerialException as E:
            raise RuntimeError(E.args)

        self.ctrl = ESP32Control(self.ser)
        self.scope = scope

    def reset(self):
        self.ctrl.reset()


    def _dis(self):
        self.ctrl.close()
        self.scope = None

    def readOutput(self):
        return np.frombuffer(self.ctrl.read_ciphertext(len(self.last_key)), \
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