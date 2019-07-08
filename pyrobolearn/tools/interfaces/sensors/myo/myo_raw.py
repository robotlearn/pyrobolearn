#!/usr/bin/env python
"""
Original by dzhu: https://github.com/dzhu/myo-raw

Edited by Fernando Cosentino: http://www.fernandocosentino.net/pyoconnect

Further cleaned by Brian Delhaisse.
"""


from __future__ import print_function

import enum
import re
import struct
import sys
import threading
import time

import serial
from serial.tools.list_ports import comports

from common import *


def multichr(ords):
    if sys.version_info[0] >= 3:
        return bytes(ords)
    else:
        return ''.join(map(chr, ords))


def multiord(b):
    if sys.version_info[0] >= 3:
        return list(b)
    else:
        return map(ord, b)


class Arm(enum.Enum):
    UNKNOWN = 0
    RIGHT = 1
    LEFT = 2


class XDirection(enum.Enum):
    UNKNOWN = 0
    X_TOWARD_WRIST = 1
    X_TOWARD_ELBOW = 2


class Pose(enum.Enum):
    REST = 0
    FIST = 1
    WAVE_IN = 2
    WAVE_OUT = 3
    FINGERS_SPREAD = 4
    THUMB_TO_PINKY = 5
    UNKNOWN = 255	 


class Packet(object):
    def __init__(self, ords):
        self.typ = ords[0]
        self.cls = ords[2]
        self.cmd = ords[3]
        self.payload = multichr(ords[4:])

    def __repr__(self):
        return 'Packet(%02X, %02X, %02X, [%s])' % \
            (self.typ, self.cls, self.cmd,
             ' '.join('%02X' % b for b in multiord(self.payload)))


class BT(object):
    """Implements the non-Myo-specific details of the Bluetooth protocol."""

    def __init__(self, tty):
        self.ser = serial.Serial(port=tty, baudrate=9600, dsrdtr=1)
        self.buf = []
        self.lock = threading.Lock()
        self.handlers = []

    # internal data-handling methods
    def recv_packet(self, timeout=None):
        """Receive a packet and call the event handlers."""
        t0 = time.time()
        self.ser.timeout = None
        while timeout is None or time.time() < t0 + timeout:
            if timeout is not None:
                self.ser.timeout = t0 + timeout - time.time()
            c = self.ser.read()
            if not c:
                return None

            ret = self.proc_byte(ord(c))
            if ret:
                if ret.typ == 0x80:
                    self.handle_event(ret)
                return ret

    def recv_packets(self, timeout=.5):
        """Receive multiple packets until the specified timeout."""
        res = []
        t0 = time.time()
        while time.time() < t0 + timeout:
            p = self.recv_packet(t0 + timeout - time.time())
            if not p:
                return res
            res.append(p)
        return res

    def proc_byte(self, c):
        """Process the received bytes."""
        if not self.buf:
            if c in [0x00, 0x80, 0x08, 0x88]:
                self.buf.append(c)
            return None
        elif len(self.buf) == 1:
            self.buf.append(c)
            self.packet_len = 4 + (self.buf[0] & 0x07) + self.buf[1]
            return None
        else:
            self.buf.append(c)

        if self.packet_len and len(self.buf) == self.packet_len:
            p = Packet(self.buf)
            self.buf = []
            return p
        return None

    def handle_event(self, p):
        """Send the processed received packet to the event/data handlers."""
        for h in self.handlers:
            h(p)

    def add_handler(self, h):
        """Add an event handler to the list of event/data handlers"""
        self.handlers.append(h)

    def remove_handler(self, h):
        """Try to remove the first instance of the specified event/data handler"""
        try:
            self.handlers.remove(h)
        except ValueError:
            pass

    def wait_event(self, cls, cmd):
        """Wait for an event"""
        res = [None]

        def h(p):
            if p.cls == cls and p.cmd == cmd:
                res[0] = p

        self.add_handler(h)
        while res[0] is None:
            self.recv_packet()
        self.remove_handler(h)
        return res[0]

    # specific BLE commands
    def connect(self, addr):
        return self.send_command(6, 3, pack('6sBHHHH', multichr(addr), 0, 6, 6, 64, 0))

    def get_connections(self):
        return self.send_command(0, 6)

    def discover(self):
        return self.send_command(6, 2, b'\x01')

    def end_scan(self):
        return self.send_command(6, 4)

    def disconnect(self, h):
        return self.send_command(3, 0, pack('B', h))

    def read_attr(self, con, attr):
        self.send_command(4, 4, pack('BH', con, attr))
        return self.wait_event(4, 5)

    def write_attr(self, con, attr, val):
        self.send_command(4, 5, pack('BHB', con, attr, len(val)) + val)
        return self.wait_event(4, 1)

    def send_command(self, cls, cmd, payload=b'', wait_resp=True):
        s = pack('4B', 0, len(payload), cls, cmd) + payload
        self.ser.write(s)

        while True:
            p = self.recv_packet()

            # no timeout, so p won't be None
            if p.typ == 0:
                return p

            # not a response: must be an event
            self.handle_event(p)


class MyoRaw(object):
    """Implements the Myo-specific communication protocol."""

    def __init__(self, tty=None, verbose=False):
        """
        Initialize the MyoRaw instance.

        Args:
            tty (str, None): tty (None, str): TTY (on Linux). You can check the list on a terminal by typing
                `ls /dev/tty*`. By default, it will be '/dev/ttyACM0'.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        if tty is None:
            tty = self.detect_tty()
        if tty is None:
            raise ValueError('Myo dongle not found!')

        self.verbose = verbose

        self.bt = BT(tty)
        self.conn = None
        self.emg_handlers = []
        self.imu_handlers = []
        self.arm_handlers = []
        self.pose_handlers = []

    def detect_tty(self):
        """Detect automatically the tty to connect to the Myo armband."""
        for p in comports():
            if re.search(r'PID=2458:0*1', p[2]):
                if self.verbose:
                    print('using device: {}'.format(p[0]))
                return p[0]

        return None

    def run(self, timeout=None):
        """Run the MyoRaw. This has to be called at each time step."""
        self.bt.recv_packet(timeout)

    def connect(self):
        """Connect to the Myo armband."""
        # stop everything from before
        self.bt.end_scan()
        self.bt.disconnect(0)
        self.bt.disconnect(1)
        self.bt.disconnect(2)

        # start scanning
        if self.verbose:
            print('scanning...')
        self.bt.discover()
        while True:
            p = self.bt.recv_packet()
            if self.verbose:
                print('scan response: {}'.format(p))

            if p.payload.endswith(b'\x06\x42\x48\x12\x4A\x7F\x2C\x48\x47\xB9\xDE\x04\xA9\x01\x00\x06\xD5'):
                addr = list(multiord(p.payload[2:8]))
                break
        self.bt.end_scan()

        # connect and wait for status event
        conn_pkt = self.bt.connect(addr)
        self.conn = multiord(conn_pkt.payload)[-1]
        self.bt.wait_event(3, 0)

        # get firmware version
        fw = self.read_attr(0x17)
        _, _, _, _, v0, v1, v2, v3 = unpack('BHBBHHHH', fw.payload)
        if self.verbose:
            print('firmware version: %d.%d.%d.%d' % (v0, v1, v2, v3))

        self.old = (v0 == 0)

        if self.old:
            # don't know what these do; Myo Connect sends them, though we get data fine without them
            self.write_attr(0x19, b'\x01\x02\x00\x00')
            self.write_attr(0x2f, b'\x01\x00')
            self.write_attr(0x2c, b'\x01\x00')
            self.write_attr(0x32, b'\x01\x00')
            self.write_attr(0x35, b'\x01\x00')

            # enable EMG data
            self.write_attr(0x28, b'\x01\x00')
            # enable IMU data
            self.write_attr(0x1d, b'\x01\x00')

            # Sampling rate of the underlying EMG sensor, capped to 1000. If it's
            # less than 1000, emg_hz is correct. If it is greater, the actual
            # framerate starts dropping inversely. Also, if this is much less than
            # 1000, EMG data becomes slower to respond to changes. In conclusion,
            # 1000 is probably a good value.
            C = 1000
            emg_hz = 50
            # strength of low-pass filtering of EMG data
            emg_smooth = 100

            imu_hz = 50

            # send sensor parameters, or we don't get any data
            self.write_attr(0x19, pack('BBBBHBBBBB', 2, 9, 2, 1, C, emg_smooth, C // emg_hz, imu_hz, 0, 0))

        else:
            name = self.read_attr(0x03)
            if self.verbose:
                print('device name: %s' % name.payload)

            # enable IMU data
            self.write_attr(0x1d, b'\x01\x00')
            # enable on/off arm notifications
            self.write_attr(0x24, b'\x02\x00')

            # self.write_attr(0x19, b'\x01\x03\x00\x01\x01')
            self.start_raw()

        # add data handlers (which are called each time we receive a packet)
        def handle_data(p):
            if (p.cls, p.cmd) != (4, 5):
                return

            c, attr, typ = unpack('BHB', p.payload[:4])
            pay = p.payload[5:]

            if attr == 0x27:  # EMG
                vals = unpack('8HB', pay)
                # not entirely sure what the last byte is, but it's a bitmask that
                # seems to indicate which sensors think they're being moved around or
                # something
                emg = vals[:8]
                moving = vals[8]
                print("EEEEMMMMMMGGGGGG: ", emg)
                self.on_emg(emg, moving)

            elif attr == 0x1c:  # IMU
                vals = unpack('10h', pay)
                quat = vals[:4]
                acc = vals[4:7]
                gyro = vals[7:10]
                self.on_imu(quat, acc, gyro)

            elif attr == 0x23:  # Arm + Pose
                typ, val, xdir, _, _, _ = unpack('6B', pay)

                if typ == 1:  # on arm
                    self.on_arm(Arm(val), XDirection(xdir))
                elif typ == 2:  # removed from arm
                    self.on_arm(Arm.UNKNOWN, XDirection.UNKNOWN)
                elif typ == 3:  # pose
                    self.on_pose(Pose(val))
            else:
                if self.verbose:
                    print('data with unknown attr: %02X %s' % (attr, p))

        # add the data handler to the list of handlers processed by the Bluetooth
        self.bt.add_handler(handle_data)

    def write_attr(self, attr, val):
        """Write the given value to the specified attribute."""
        if self.conn is not None:
            self.bt.write_attr(self.conn, attr, val)

    def read_attr(self, attr):
        """Read the specified attribute."""
        if self.conn is not None:
            return self.bt.read_attr(self.conn, attr)
        return None

    def disconnect(self):
        """Disconnect from the myo armband."""
        if self.conn is not None:
            self.bt.disconnect(self.conn)

    def start_raw(self):
        """Sending this sequence for v1.0 firmware seems to enable both raw data and
        pose notifications.
        """

        self.write_attr(0x28, b'\x01\x00')  # EMG?
        # self.write_attr(0x19, b'\x01\x03\x01\x01\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')

    def mc_start_collection(self):
        """Myo Connect sends this sequence (or a reordering) when starting data
        collection for v1.0 firmware; this enables raw data but disables arm and
        pose notifications.
        """

        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')
        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x19, b'\x09\x01\x01\x00\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x19, b'\x01\x03\x00\x01\x00')
        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x00')

    def mc_end_collection(self):
        """Myo Connect sends this sequence (or a reordering) when ending data collection
        for v1.0 firmware; this reenables arm and pose notifications, but
        doesn't disable raw data.
        """

        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')
        self.write_attr(0x19, b'\x09\x01\x00\x00\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x00\x01\x01')
        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')

    def vibrate(self, length):
        """
        Vibrate the myo armband for the specified length.

        Args:
            length (int): integer between 1 and 4.
        """
        if length in xrange(1, 4):
            # first byte tells it to vibrate; purpose of second byte is unknown
            self.write_attr(0x19, pack('3B', 3, 1, length))

    def add_emg_handler(self, h):
        """Add an EMG handler function/method which will be called each time we receive a packet. The handler function
        has to accept as inputs two parameters where the first one will be a list of 8 EMG sensor values, and the
        other will be an int to specify if it is moving or not.

        Args:
            h (callable): EMG handler function.
        """
        self.emg_handlers.append(h)

    def add_imu_handler(self, h):
        """Add an IMU handler function/method which will be called each time we receive a packet. The handler function
        has to accept as inputs 3 parameters where the first one will be the orientation expressed as a quaternion,
        the 3 acceleration values returned by the accelerometer, and the 3 angular velocity values returned by the
        gyro.

        Args:
            h (callable): IMU handler function.
        """
        self.imu_handlers.append(h)

    def add_pose_handler(self, h):
        """Add a Pose handler function/method which will be called each time we receive a packet. The handler function
        has to accept as input one parameter which is the pose (an instance of ``Pose``).

        Args:
            h (callable): pose handler function.
        """
        self.pose_handlers.append(h)

    def add_arm_handler(self, h):
        """
        Add an Arm handler function/method which will be called each time we receive a packet. The handler function
        has to accept as inputs two parameters where the first one will be the arm (an instance of ``Arm``), and
        the second will be the x direction (an instance of ``XDirection``).

        Args:
            h (callable): arm handler function.
        """
        self.arm_handlers.append(h)

    def on_emg(self, emg, moving):
        """
        Call each EMG handler function, and pass them the EMG values and if moving or not.
        """
        for h in self.emg_handlers:
            h(emg, moving)

    def on_imu(self, quat, acc, gyro):
        """
        Call each IMU handler function, and pass them the IMU values (quaternion, acceleration, gyro).
        """
        for h in self.imu_handlers:
            h(quat, acc, gyro)

    def on_pose(self, p):
        """
        Call each Pose handler function, and pass them the Pose instance.
        """
        for h in self.pose_handlers:
            h(p)

    def on_arm(self, arm, xdir):
        """
        Call each Arm handler function, and pass them the Arm and XDirection instances.
        """
        for h in self.arm_handlers:
            h(arm, xdir)


if __name__ == '__main__':
    try:
        import pygame
        from pygame.locals import *
        HAVE_PYGAME = True
    except ImportError:
        HAVE_PYGAME = False

    if HAVE_PYGAME:
        w, h = 1200, 400
        scr = pygame.display.set_mode((w, h))

    last_vals = None

    def plot(scr, vals):
        DRAW_LINES = False

        global last_vals
        if last_vals is None:
            last_vals = vals
            return

        D = 5
        scr.scroll(-D)
        scr.fill((0,0,0), (w - D, 0, w, h))
        for i, (u, v) in enumerate(zip(last_vals, vals)):
            if DRAW_LINES:
                pygame.draw.line(scr, (0,255,0),
                                 (w - D, int(h/8 * (i+1 - u))),
                                 (w, int(h/8 * (i+1 - v))))
                pygame.draw.line(scr, (255,255,255),
                                 (w - D, int(h/8 * (i+1))),
                                 (w, int(h/8 * (i+1))))
            else:
                c = int(255 * max(0, min(1, v)))
                scr.fill((c, c, c), (w - D, i * h / 8, D, (i + 1) * h / 8 - i * h / 8));

        pygame.display.flip()
        last_vals = vals

    m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)

    def proc_emg(emg, moving, times=[]):
        if HAVE_PYGAME:
            # update pygame display
            plot(scr, [e / 2000. for e in emg])
        else:
            print(emg)

        # print framerate of received data
        times.append(time.time())
        if len(times) > 20:
            # print((len(times) - 1) / (times[-1] - times[0]))
            times.pop(0)

    m.add_emg_handler(proc_emg)
    m.connect()

    m.add_arm_handler(lambda arm, xdir: print('arm', arm, 'xdir', xdir))
    m.add_pose_handler(lambda p: print('pose', p))

    try:
        while True:
            m.run(1)

            if HAVE_PYGAME:
                for ev in pygame.event.get():
                    if ev.type == QUIT or (ev.type == KEYDOWN and ev.unicode == 'q'):
                        raise KeyboardInterrupt()
                    elif ev.type == KEYDOWN:
                        if K_1 <= ev.key <= K_3:
                            m.vibrate(ev.key - K_0)
                        if K_KP1 <= ev.key <= K_KP3:
                            m.vibrate(ev.key - K_KP0)

    except KeyboardInterrupt:
        pass
    finally:
        m.disconnect()
        print()
