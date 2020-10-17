# PySerial Emulator

This code is used to show examples of how you would use PySerial to interact with an Arduino. It emulates the Arduino as a source to read from and also sends data to the emulated Arduino.

## Setup your pipenv

From the folder with the Pipfile run the command `pipenv update`

If you don't have pipenv installed please follow the instructions at https://pypi.org/project/pipenv/

## Running the code

The code can be run with the following command:
* `python3 testSerialSimulator.py`

## Expected output
```
0
Arduino got: "My car drives itself"
read one byte =  W
read multiple bytes =  e used to
read a line =  drive cars.

Serial port information...
    Serial<id=0xa81c10, open=<bound method Serial.isOpen of <fakeSerial.Serial object at 0x10dcb5af0>>>( port='0', baudrate=9600, bytesize=8, parity='N', stopbits=1, xonxoff=0, rtscts=0)
Serial port open = True
Serial port open = False
```
