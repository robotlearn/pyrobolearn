# uncompyle6 version 3.3.5
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.12 (default, Nov 12 2018, 14:36:49) 
# [GCC 5.4.0 20160609]
# Embedded file name: PyoConnectLib.py
# Compiled at: 2015-08-05 16:17:41
"""
    PyoConnect v0.1

    Author:
      Fernando Cosentino - fbcosentino@yahoo.com.br

    Official source:
      http://www.fernandocosentino.net/pyoconnect

    Based on the work of dzhu: https://github.com/dzhu/myo-raw

    License:
            Use at will, modify at will. Always keep my name in this file as original author. And that's it.

    Steps required (in a clean debian installation) to use this library:
            // permission to ttyACM0 - must logout and login again on Linux
            sudo usermod -a -G dialout $USER

            // dependencies
            apt-get install python-pip
            pip install pySerial --upgrade
            pip install enum34
            pip install PyUserInput
            apt-get install python-Xlib

            // now logout and login again

Note that this file has been modified (mostly cleaned) with respect to the original by Brian Delhaisse.
"""
from __future__ import print_function
import sys
import time
from subprocess import Popen, PIPE
import re
import math

try:
    from pymouse import PyMouse
    pmouse = PyMouse()
except:
    print('PyMouse error: No mouse support')
    pmouse = None
else:
    try:
        from pykeyboard import PyKeyboard
        pkeyboard = PyKeyboard()
    except:
        print('PyKeyboard error: No keyboard support')
        pkeyboard = None

# from common import *
from myo_raw import MyoRaw, Pose, Arm, XDirection


class Myo(MyoRaw):
    """Myo class"""

    def __init__(self, tty=None, verbose=False):
        """
        Initialize the Myo instance.

        Args:
            tty (str, None): tty (None, str): TTY (on Linux). You can check the list on a terminal by typing
                `ls /dev/tty*`. By default, it will be '/dev/ttyACM0'.
            verbose (bool): If True, it will print information about the state of the interface. This is let to the
                programmer what he / she wishes to print.
        """
        self.verbose = verbose
        self.locked = True
        self.use_lock = True
        self.timed = True
        self.lock_time = 1.0
        self.time_to_lock = self.lock_time
        self.last_pose = -1
        self.last_tick = 0
        self.current_box = 0
        self.last_box = 0
        self.box_factor = 0.25
        self.current_arm = 0
        self.current_xdir = 0
        self.current_gyro = None
        self.current_accel = None
        self.current_roll = 0
        self.current_pitch = 0
        self.current_yaw = 0
        self.center_roll = 0
        self.center_pitch = 0
        self.center_yaw = 0
        self.first_rot = 0
        self.current_rot_roll = 0
        self.current_rot_pitch = 0
        self.current_rot_yaw = 0
        self.mov_history = ''
        self.gest_history = ''
        self.act_history = ''
        if pmouse is not None:
            self.x_dim, self.y_dim = pmouse.screen_size()
            self.mx = self.x_dim / 2
            self.my = self.y_dim / 2
        self.centered = 0

        self.current_emg_values = []
        self.bitmask_moving = 0

        MyoRaw.__init__(self, tty=tty, verbose=verbose)
        self.add_emg_handler(self.emg_handler)
        self.add_arm_handler(self.arm_handler)
        self.add_imu_handler(self.imu_handler)
        self.add_pose_handler(self.pose_handler)
        self.onEMG = None
        self.onPoseEdge = None
        self.onPoseEdgeList = []
        self.onLock = None
        self.onLockList = []
        self.onUnlock = None
        self.onUnlockList = []
        self.onPeriodic = None
        self.onPeriodicList = []
        self.onWear = None
        self.onWearList = []
        self.onUnwear = None
        self.onUnwearList = []
        self.onBoxChange = None
        self.onBoxChangeList = []
        return

    def check_myo_around(self):
        self.bt.end_scan()
        self.bt.disconnect(0)
        self.bt.disconnect(1)
        self.bt.disconnect(2)
        self.bt.discover()
        p = self.bt.recv_packet(1)
        try:
            pl = p.payload
        except:
            pl = ''

        if pl.endswith('\x06BH\x12J\x7f,HG\xb9\xde\x04\xa9\x01\x00\x06\xd5'):
            self.bt.end_scan()
            return True
        else:
            return False

    def tick(self):
        now = time.time()
        if now - self.last_tick >= 0.01:
            if self.onPeriodic is not None:
                self.onPeriodic()
            for h in self.onPeriodicList:
                h()

            if self.use_lock and self.locked == False and self.timed:
                if self.time_to_lock <= 0:
                    if self.verbose:
                        print('Locked')
                    self.locked = True
                    self.vibrate(1)
                    self.time_to_lock = self.lock_time
                    if self.onLock is not None:
                        self.onLock()
                    for h in self.onLockList:
                        h()

                else:
                    self.time_to_lock -= 0.01
            self.last_tick = now
        return

    def clear_handle_lists(self):
        self.onPoseEdgeList = []
        self.onLockList = []
        self.onUnlockList = []
        self.onPeriodicList = []
        self.onWearList = []
        self.onUnwearList = []
        self.onBoxChangeList = []
        self.emg_handlers = []

    def Add_onPoseEdge(self, h):
        self.onPoseEdgeList.append(h)

    def Add_onLock(self, h):
        self.onLockList.append(h)

    def Add_onUnlock(self, h):
        self.onUnlockList.append(h)

    def Add_onPeriodic(self, h):
        self.onPeriodicList.append(h)

    def Add_onWear(self, h):
        self.onWearList.append(h)

    def Add_onUnwear(self, h):
        self.onUnwearList.append(h)

    def Add_onBoxChange(self, h):
        self.onBoxChangeList.append(h)

    def emg_handler(self, emg, moving):
        """EMG handler."""
        # if self.onEMG is not None:
        #     self.onEMG(emg, moving)
        self.current_emg_values = emg
        self.bitmask_moving = moving
        return

    def arm_handler(self, arm, xdir):
        """Arm handler."""
        if arm == Arm(0):
            self.current_arm = 'unknown'
        elif arm == Arm(1):
            self.current_arm = 'right'
        elif arm == Arm(2):
            self.current_arm = 'left'
        if xdir == XDirection(0):
            self.current_xdir = 'unknown'
        elif xdir == XDirection(1):
            self.current_xdir = 'towardWrist'
        elif xdir == XDirection(2):
            self.current_xdir = 'towardElbow'
        if Arm(arm) == 0:
            if self.onUnwear is not None:
                self.onUnwear()
            for h in self.onUnwearList:
                h()

        elif self.onWear is not None:
            self.onWear(self.current_arm, self.current_xdir)
        else:
            for h in self.onWearList:
                h(self.current_arm, self.current_xdir)

        return

    def imu_handler(self, quat, acc, gyro):
        """IMU handler"""
        q0, q1, q2, q3 = quat
        q0 = q0 / 16384.0
        q1 = q1 / 16384.0
        q2 = q2 / 16384.0
        q3 = q3 / 16384.0
        self.current_roll = math.atan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
        self.current_pitch = -math.asin(max(-1.0, min(1.0, 2.0 * (q0 * q2 - q3 * q1))))
        self.current_yaw = -math.atan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
        self.current_rot_roll = self.angle_dif(self.current_roll, self.center_roll)
        self.current_rot_yaw = self.angle_dif(self.current_yaw, self.center_yaw)
        self.current_rot_pitch = self.angle_dif(self.current_pitch, self.center_pitch)
        g0, g1, g2 = gyro
        g0 = g0 / 16.0
        g1 = g1 / 16.0
        g2 = g2 / 16.0
        self.current_gyro = (g0, g1, g2)
        ac0, ac1, ac2 = acc
        ac0 = ac0 / 2048.0
        ac1 = ac1 / 2048.0
        ac2 = ac2 / 2048.0
        self.current_accel = (ac0, ac1, ac2)
        if self.first_rot == 0:
            self.rotSetCenter()
            self.first_rot = 1
        self.current_box = self.getBox()
        if self.current_box != self.last_box:
            self.mov_history = str(self.mov_history[-100:]) + str(self.current_box)
            self.act_history = str(self.act_history[-100:]) + str(self.current_box)
            if self.onBoxChange is not None:
                self.onBoxChange(self.last_box, 'off')
                self.onBoxChange(self.current_box, 'on')
            for h in self.onBoxChangeList:
                h(self.last_box, 'off')
                h(self.current_box, 'on')

            self.last_box = self.current_box
        return

    def pose_handler(self, p):
        """Pose handler."""
        if p == Pose(0):
            pn = 0
        elif p == Pose(1):
            pn = 1
        elif p == Pose(2):
            pn = 2
        elif p == Pose(3):
            pn = 3
        elif p == Pose(4):
            pn = 4
        elif p == Pose(5):
            pn = 5
        else:
            pn = 6
        if pn != self.last_pose:
            self.gest_history = str(self.gest_history[-100:]) + str(self.PoseToChar(pn))
            self.act_history = str(self.act_history[-100:]) + str(self.PoseToChar(pn))
            if self.locked == False:
                self.time_to_lock = self.lock_time
                if self.last_pose > -1:
                    if self.onPoseEdge is not None:
                        self.onPoseEdge(self.PoseToStr(self.last_pose), 'off')
                    for h in self.onPoseEdgeList:
                        h(self.PoseToStr(pn), 'off')

                if self.onPoseEdge is not None:
                    self.onPoseEdge(self.PoseToStr(pn), 'on')
                for h in self.onPoseEdgeList:
                    h(self.PoseToStr(pn), 'on')

            self.last_pose = pn
        if pn == 5 and self.locked and self.use_lock:
            self.locked = False
            self.vibrate(1)
            if self.verbose:
                print('unlock')
            if self.onUnlock is not None:
                self.onUnlock()
            for h in self.onUnlockList:
                h()

        return

    @property
    def arm(self):
        """Get the arm."""
        return self.current_arm

    @property
    def x_direction(self):
        return self.current_xdir

    @property
    def gyro(self):
        return self.current_gyro

    @property
    def accel(self):
        return self.current_accel

    @property
    def time_milliseconds(self):
        return round(time.time() * 1000)

    @property
    def roll(self):
        return self.current_roll

    @property
    def pitch(self):
        return self.current_pitch

    @property
    def yaw(self):
        return self.current_yaw

    def setLockingPolicy(self, policy):
        if policy == 'none':
            self.use_lock = False
        elif policy == 'standard':
            self.use_lock = True

    def lock(self):
        self.locked = True
        self.vibrate(1)
        if self.onLock is not None:
            self.onLock()
        for h in self.onLockList:
            h()

        return

    def unlock(self, unlock_type):
        if unlock_type == 'timed':
            self.vibrate(1)
            self.locked = False
            self.timed = True
        if unlock_type == 'hold':
            self.vibrate(1)
            self.locked = False
            self.timed = False

    def isUnlocked(self):
        if self.locked:
            return False
        else:
            return True

    def notifyUserAction(self):
        self.vibrate(1)

    def keyboard(self, kkey, kedge, kmod):
        if pkeyboard is not None:
            tkey = kkey
            if tkey == 'left_arrow':
                tkey = pkeyboard.left_key
            if tkey == 'right_arrow':
                tkey = pkeyboard.right_key
            if tkey == 'up_arrow':
                tkey = pkeyboard.up_key
            if tkey == 'down_arrow':
                tkey = pkeyboard.down_key
            if tkey == 'space':
                pass
            if tkey == 'return':
                tkey = pkeyboard.return_key
            if tkey == 'escape':
                tkey = pkeyboard.escape_key
            if kmod == 'left_shift':
                pkeyboard.press_key(pkeyboard.shift_l_key)
            if kmod == 'right_shift':
                pkeyboard.press_key(pkeyboard.shift_r_key)
            if kmod == 'left_control':
                pkeyboard.press_key(pkeyboard.control_l_key)
            if kmod == 'right_control':
                pkeyboard.press_key(pkeyboard.control_r_key)
            if kmod == 'left_alt':
                pkeyboard.press_key(pkeyboard.alt_l_key)
            if kmod == 'right_alt':
                pkeyboard.press_key(pkeyboard.alt_r_key)
            if kmod == 'left_win':
                pkeyboard.press_key(pkeyboard.super_l_key)
            if kmod == 'right_win':
                pkeyboard.press_key(pkeyboard.super_r_key)
            if kedge == 'down':
                pkeyboard.press_key(tkey)
            elif kedge == 'up':
                pkeyboard.release_key(tkey)
            elif kedge == 'press':
                pkeyboard.tap_key(tkey)
            if kmod == 'left_shift':
                pkeyboard.release_key(pkeyboard.shift_l_key)
            if kmod == 'right_shift':
                pkeyboard.release_key(pkeyboard.shift_r_key)
            if kmod == 'left_control':
                pkeyboard.release_key(pkeyboard.control_l_key)
            if kmod == 'right_control':
                pkeyboard.release_key(pkeyboard.control_r_key)
            if kmod == 'left_alt':
                pkeyboard.release_key(pkeyboard.alt_l_key)
            if kmod == 'right_alt':
                pkeyboard.release_key(pkeyboard.alt_r_key)
            if kmod == 'left_win':
                pkeyboard.release_key(pkeyboard.super_l_key)
            if kmod == 'right_win':
                pkeyboard.release_key(pkeyboard.super_r_key)
        return

    def centerMousePosition(self):
        if pmouse is not None:
            x_dim, y_dim = pmouse.screen_size()
            pmouse.move(x_dim / 2, y_dim / 2)
        return

    def mouse(self, button, edge, mod):
        if pmouse is not None:
            mpos = pmouse.position()
            if button == 'left':
                mbut = 1
            elif button == 'right':
                mbut = 2
            elif button == 'center':
                mbut = 3
            else:
                mbut = 1
            if edge == 'down':
                pmouse.press(mpos[0], mpos[1], mbut)
            elif edge == 'up':
                pmouse.release(mpos[0], mpos[1], mbut)
            elif edge == 'click':
                pmouse.click(mpos[0], mpos[1], mbut)
        return

    @property
    def pose(self):
        return self.PoseToStr(self.last_pose)

    def getPoseSide(self):
        if self.last_pose == 2 and self.current_arm == 'right' or self.last_pose == 3 and self.current_arm == 'left':
            return 'waveLeft'
        if self.last_pose == 3 and self.current_arm == 'right' or self.last_pose == 2 and self.current_arm == 'left':
            return 'waveRight'
        return self.PoseToStr(self.last_pose)

    def isLocked(self):
        return self.locked

    def mouseMove(self, x, y):
        if pmouse is not None:
            pmouse.move(x, y)
        return

    def title_contains(self, text):
        window_str = self.get_active_window_title()
        if window_str.find(text) > -1:
            return True
        else:
            return False

    def class_contains(self, text):
        window_str = self.get_active_window_class()
        if window_str.find(text) > -1:
            return True
        else:
            return False

    def rotSetCenter(self):
        self.center_roll = self.current_roll
        self.center_pitch = self.current_pitch
        self.center_yaw = self.current_yaw

    def rotRoll(self):
        return self.current_rot_roll

    def rotPitch(self):
        return self.current_rot_pitch

    def rotYaw(self):
        return self.angle_dif(self.current_yaw, self.center_yaw)

    def getBox(self):
        if self.current_rot_pitch > self.box_factor:
            if self.current_rot_yaw > self.box_factor:
                return 2
            else:
                if self.current_rot_yaw < -self.box_factor:
                    return 8
                return 1

        elif self.current_rot_pitch < -self.box_factor:
            if self.current_rot_yaw > self.box_factor:
                return 4
            else:
                if self.current_rot_yaw < -self.box_factor:
                    return 6
                return 5

        elif self.current_rot_yaw > self.box_factor:
            return 3
        else:
            if self.current_rot_yaw < -self.box_factor:
                return 7
            return 0

    def getHBox(self):
        if self.current_rot_yaw > self.box_factor:
            return 1
        else:
            if self.current_rot_yaw < -self.box_factor:
                return -1
            return 0

    def getVBox(self):
        if self.current_rot_pitch > self.box_factor:
            return 1
        else:
            if self.current_rot_pitch < -self.box_factor:
                return -1
            return 0

    def clearHistory(self):
        self.mov_history = ''
        self.gest_history = ''
        self.act_history = ''

    def getLastMovements(self, num):
        if num >= 0:
            return self.mov_history[-num:]
        else:
            return self.mov_history

    def getLastGestures(self, num):
        if num >= 0:
            return self.gest_history[-num:]
        else:
            return self.gest_history

    def getLastActions(self, num):
        if num >= 0:
            return self.act_history[-num:]
        else:
            return self.act_history

    def PoseToStr(self, posenum):
        """Return a string describing the given pose id.

        Check: https://support.getmyo.com/hc/article_attachments/115012009363/myo-for-education-lesson-5.pdf
        """
        if posenum == 0:
            return 'rest'
        else:
            if posenum == 1:
                return 'fist'
            if posenum == 2:
                return 'waveIn'
            if posenum == 3:
                return 'waveOut'
            if posenum == 4:
                return 'fingersSpread'
            if posenum == 5:
                return 'doubleTap'
            return 'unknown'

    def PoseToChar(self, posenum):
        """Return a char describing the given pose id.

        Here is what each letter stands for:
        'R' --> rest
        'F' --> fist
        'I' --> wave in
        'O' --> wave out
        'S' --> fingers spread
        'D' --> double tap
        'U' --> unknown

        Check: https://support.getmyo.com/hc/article_attachments/115012009363/myo-for-education-lesson-5.pdf
        """
        if posenum == 0:
            return 'R'
        else:
            if posenum == 1:
                return 'F'
            if posenum == 2:
                return 'I'
            if posenum == 3:
                return 'O'
            if posenum == 4:
                return 'S'
            if posenum == 5:
                return 'D'
            return 'U'

    def limit_angle(self, angle):
        if angle > math.pi:
            return angle - 2.0 * math.pi
        if angle < -2.0 * math.pi:
            return angle + 2.0 * math.pi
        return angle

    def angle_dif(self, angle, ref):
        if ref >= 0:
            if angle >= 0:
                return self.limit_angle(angle - ref)
            else:
                if angle >= ref - math.pi:
                    return self.limit_angle(angle - ref)
                return self.limit_angle(angle + 2.0 * math.pi - ref)

        elif angle <= 0:
            return self.limit_angle(angle - ref)
        else:
            if angle <= ref + math.pi:
                return self.limit_angle(angle - ref)
            return self.limit_angle(angle - 2.0 * math.pi - ref)

    def get_active_window_title(self):
        try:
            root = Popen(['xprop', '-root', '_NET_ACTIVE_WINDOW'], stdout=PIPE)
            for line in root.stdout:
                mw = re.search('^_NET_ACTIVE_WINDOW.* ([\\w]+)$', line)
                if mw is not None:
                    id_ = mw.group(1)
                    id_w = Popen(['xprop', '-id', id_, 'WM_NAME'], stdout=PIPE)
                    break

            if id_w is not None:
                for line in id_w.stdout:
                    match = re.match('WM_NAME\\(\\w+\\) = (?P<name>.+)$', line)
                    if match is not None:
                        return match.group('name')

            return ''
        except:
            return ''

        return

    def get_active_window_class(self):
        try:
            root = Popen(['xprop', '-root', '_NET_ACTIVE_WINDOW'], stdout=PIPE)
            for line in root.stdout:
                mw = re.search('^_NET_ACTIVE_WINDOW.* ([\\w]+)$', line)
                if mw is not None:
                    id_ = mw.group(1)
                    id_w = Popen(['xprop', '-id', id_, 'WM_CLASS'], stdout=PIPE)
                    break

            if id_w is not None:
                for line in id_w.stdout:
                    match = re.match('WM_CLASS\\(\\w+\\) = (?P<name>.+)$', line)
                    if match is not None:
                        return match.group('name')

            return ''
        except:
            return ''

        return


if __name__ == '__main__':
    tty = sys.argv[1] if len(sys.argv) >= 2 else None
    m = Myo(tty=tty, verbose=True)
    m.connect()

    while True:
        m.run()
