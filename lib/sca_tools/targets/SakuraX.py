from chipwhisperer.capture.targets._base import TargetTemplate
from Crypto.Cipher import AES

try:
    import ftd2xx
    ftd2xx_available = True
except OSError:
    ftd2xx_available = False

import time

DEVICE_NAME = "FT8P1RTUA"

class SakuraXControl:
    KEY_ADDR = 0x100
    PLAINTEXT_ADDR = 0x140
    CIPHERTEXT_ADDR = 0x180
    KICK_ADDR = 0x2
    def __init__(self, dev) -> None:
        self.dev = dev
        self.dev.setBaudRate(115200)
        self.dev.setTimeouts(1000, 1000)

    def write_data(self, addr : int, data : bytes):
        # data must be 2 bytes
        # cmd
        b = 0x01.to_bytes(1, "big")
        # addr
        addr_bytes = addr.to_bytes(2, "big")
        b += addr_bytes
        # data
        b += data
        self.dev.write(b)

    def read_data(self, addr : int):
        # cmd
        b = 0x00.to_bytes(1, "big")
        addr_bytes = addr.to_bytes(2, "big")
        b += addr_bytes
        self.dev.write(b)
        ret = self.dev.read(2)
        if len(ret) < 2:
            raise Exception("Timeout Error")
        return ret

    def flush(self):
        # wait until rx buffer is empty
        while self.dev.getQueueStatus()[0] != 0:
            time.usleep(100)

    def reset(self):
        self.dev.resetDevice()
        self.dev.purge(3) # purge rx(0x1) and tx(0x2) buffer

    def close(self):
        self.dev.close()

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
        if not ftd2xx_available:
            raise RuntimeError("ftd2xx is not available")
        self.connectStatus = False
        self.ctrl = None
        self.scope = None
        self.cipher = None
        self.last_key = [0 for _ in range(16)]

    def getName(self):
        return self._name

    # implimentation of con method
    def _con(self, scope):
        """Connect to SAKURA-X Controller"""
        try:
            dev_list = [dev.decode("utf-8") for dev in ftd2xx.listDevices()]
            idx = dev_list.index(DEVICE_NAME)
        except (ValueError,TypeError):
            raise RuntimeError("SAKURA-X Controller not found")
        self.ctrl = SakuraXControl(ftd2xx.open(idx))

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
        return self.ctrl.read_ciphertext(len(self.last_key))

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