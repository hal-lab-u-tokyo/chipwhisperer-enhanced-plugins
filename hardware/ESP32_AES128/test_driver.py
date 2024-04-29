import serial
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# Serial port settings
port = "/dev/ttyUSB4"
baudrate = 115200

# Serial port open
ser = serial.Serial(port, baudrate)

CMD_SET_KEY 		= 0x11
CMD_SET_PLAINTEXT	= 0x12
CMD_ENCRYPT			= 0x13
CMD_GET_CIPHERTEXT	= 0x14
CMD_GET_DEBUG		= 0x15

# Send data
def send_data(data):
    ser.write(data.encode())

# Receive data
def read_data():
    while True:
        if ser.in_waiting > 0:
            incoming_data = ser.readline()
            print("Received:", incoming_data.decode().strip())

def set_key(key):
    buf = b""
    buf += CMD_SET_KEY.to_bytes(1, 'big')
    buf += key
    ser.write(buf)

def set_plaintext(plaintext):
    buf = b""
    buf += CMD_SET_PLAINTEXT.to_bytes(1, 'big')
    buf += plaintext
    ser.write(buf)

def encrypt():
    buf = b""
    buf += CMD_ENCRYPT.to_bytes(1, 'big')
    ser.write(buf)
    stat = ser.read(1)
    if stat[0] != 0:
        print("Error: encrypt failed")

def get_ciphertext():
    buf = b""
    buf += CMD_GET_CIPHERTEXT.to_bytes(1, 'big')
    ser.write(buf)
    return ser.read(16)

def get_debug():
    buf = b""
    buf += CMD_GET_DEBUG.to_bytes(1, 'big')
    ser.write(buf)
    key = ser.read(16)
    pt = ser.read(16)
    ct = ser.read(16)
    print("\nDebug info:")
    print("Key: ")
    print_hex_128bit(key)
    print("Plain text: ")
    print_hex_128bit(pt)
    print("Cipher text: ")
    print_hex_128bit(ct)


def print_hex_128bit(data):
    for b in data:
        print(f"{b:02X}", end = " ")
    print()


def prepare_data(cipher):
    data = get_random_bytes(16)
    answer = cipher.encrypt(data)
    return (data, answer)

def create_cipher():
    key = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_ECB)
    return (key, cipher)

# flush serial buffer
# ser.reset_input_buffer()

key, cipher = create_cipher()
print("Key: ")
print_hex_128bit(key)
set_key(key)

pt, correct_ct = prepare_data(cipher)
print("Plain text: ")
print_hex_128bit(pt)
set_plaintext(pt)

encrypt()

ct = get_ciphertext()
print("Cipher text: ")
print_hex_128bit(ct)
if ct == correct_ct:
    print("Correct!")
else:
    print("Correct Cipher text: ")
    print_hex_128bit(correct_ct)
    get_debug()

ser.close()
