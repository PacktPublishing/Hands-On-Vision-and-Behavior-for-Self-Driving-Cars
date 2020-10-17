# testSerialSimulator.py
# D. Thiebaut
# This program energizes the fakeSerial simulator using example code taken
# from http://pyserial.sourceforge.net/shortintro.html
# Modified by K. Korda

# import the simulator module (it should be in the same directory as this program)
import fakeSerial as serial

# Example 1  from http://pyserial.sourceforge.net/shortintro.html
def Example1():
    ser = serial.Serial(0)  # open first serial port
    print( ser.name )       # check which port was really used
    ser.write("My car drives itself")      # write a string
    ser.close()             # close port

# Example 2  from http://pyserial.sourceforge.net/shortintro.html
def Example2():
    ser = serial.Serial('/dev/ttyS1', 9600, timeout=1)
    x = ser.read()          # read one byte
    print( "read one byte = ", x )
    s = ser.read(10)        # read up to ten bytes (timeout)
    print( "read multiple bytes = ", s )
    line = ser.readline()   # read a '\n' terminated line
    ser.close()
    print( "read a line = ", line )

# Example 3  from http://pyserial.sourceforge.net/shortintro.html
def Example3():
    ser = serial.Serial()
    ser.baudrate = 9600
    ser.port = 0
    print( f"Serial port information...\n    {ser}" )

    ser.open()
    print( f"Serial port open = {str( ser.isOpen() )}" )

    ser.close()
    print( f"Serial port open = {str( ser.isOpen() )}" )


Example1()
Example2()
Example3()
