###
#   Copyright (C) 2024 The University of Tokyo
#   
#   File:          /lib/cw_plugins/targets/SakuraX.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  27-03-2024 18:15:59
#   Last Modified: 29-04-2024 19:06:44
###


from chipwhisperer.capture.targets._base import TargetTemplate
from Crypto.Cipher import AES
import numpy as np

import serial
from serial.serialutil import SerialException

class SakuraXControl:
    KEY_ADDR = 0x100
    PLAINTEXT_ADDR = 0x140
    CIPHERTEXT_ADDR = 0x180
    KICK_ADDR = 0x2
    def __init__(self, ser) -> None:
        self.ser : serial.Serial = ser

    def write_data(self, addr : int, data : bytes):
        # data must be 2 bytes
        # cmd
        b = 0x01.to_bytes(1, "big")
        # addr
        addr_bytes = addr.to_bytes(2, "big")
        b += addr_bytes
        # data
        b += data
        self.ser.write(b)

    def read_data(self, addr : int):
        # cmd
        b = 0x00.to_bytes(1, "big")
        addr_bytes = addr.to_bytes(2, "big")
        b += addr_bytes
        self.ser.write(b)
        ret = self.ser.read(2)
        if len(ret) < 2:
            raise Exception("Timeout Error")
        return ret

    def flush(self):
        self.ser.flush()

    def reset(self):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def close(self):
        self.ser.close()

    def send_key(self, key : bytes):
        for i in range(len(key) // 2):
            self.write_data(self.KEY_ADDR + 2 * i, key[2*i:2*i+2])

    def send_plaintext(self, plaintext : bytes):
        for i in range(len(plaintext) // 2):
            self.write_data(self.PLAINTEXT_ADDR + 2 * i, plaintext[2*i:2*i+2])

    def run(self):
        ctrl = 3
        self.write_data(self.KICK_ADDR, ctrl.to_bytes(2, "big"))

    def read_ciphertext(self, byte_len : int = 8):
        ct = b''
        for i in range(byte_len // 2):
            ct += self.read_data(self.CIPHERTEXT_ADDR + 2 * i)
        return ct

class SakuraX(TargetTemplate):
    _name = 'Sakura-X'
    def __init__(self):
        super().__init__()
        self.ser = None
        self.connectStatus = False
        self.ctrl = None
        self.scope = None
        self.cipher = None
        self.last_key = [0 for _ in range(16)]

    def getName(self):
        return self._name

    # implimentation of con method
    def _con(self, scope, serial_port = None, baud = 115200):
        """Connect to SAKURA-X Controller"""
        if serial_port is None:
            serial_port = "/dev/ttyUSB0"
        try:
            self.ser = serial.Serial(serial_port, baud, timeout=1)
        except SerialException as E:
            raise RuntimeError(E.args)

        self.ctrl = SakuraXControl(self.ser)

        self.scope = scope

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