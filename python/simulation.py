import ctypes, os
import numpy as np

lib_dirs = os.getenv("GAZEBO_PLUGIN_PATH").split(":")
libarm_name = ""
for dir_name in lib_dirs:
  if os.path.isfile(dir_name + "/libarm_plugin.so"):
    libarm_name = dir_name + "/libarm_plugin.so"
    break
libarm = ctypes.cdll.LoadLibrary(libarm_name)
libarm.arm_plugin_init.resType = None
libarm.arm_plugin_destroy.resType = None
libarm.arm_plugin_setPositions.resType = None
libarm.arm_plugin_setPositions.argTypes = [ \
    ctypes.c_double, ctypes.c_double, ctypes.c_double, \
    ctypes.c_double, ctypes.c_double, ctypes.c_double, \
    ctypes.c_double, ctypes.c_double, ctypes.c_double, \
    ctypes.c_double, ctypes.c_double, ctypes.c_double, \
    ctypes.c_double, ctypes.c_double, ctypes.c_double, \
    ctypes.c_double, ctypes.c_double, ctypes.c_double, \
    ctypes.c_double, ctypes.c_double, ctypes.c_double ]

class Arm(object):
  def __init__(self):
    # set the limits here (HW 1.1)
    self.joint_limits = [
        (-180.0, 180.0),
        (-180.0, 180.0),
        (-180.0, 180.0),
        (-180.0, 180.0),
        (-180.0, 180.0),
        (-180.0, 180.0),
        (-180.0, 180.0)]
    self.num_joints = len(self.joint_limits)

    self.default_length = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # open the gazebo simulation
    libarm.arm_plugin_init()

  def __del__(self):
    libarm.arm_plugin_destroy()

  def setPositions(self, pos):
    libarm.arm_plugin_setPositions(
        ctypes.c_double(pos[0, 0]),
        ctypes.c_double(pos[0, 1]),
        ctypes.c_double(pos[0, 2]),

        ctypes.c_double(pos[1, 0]),
        ctypes.c_double(pos[1, 1]),
        ctypes.c_double(pos[1, 2]),

        ctypes.c_double(pos[2, 0]),
        ctypes.c_double(pos[2, 1]),
        ctypes.c_double(pos[2, 2]),

        ctypes.c_double(pos[3, 0]),
        ctypes.c_double(pos[3, 1]),
        ctypes.c_double(pos[3, 2]),

        ctypes.c_double(pos[4, 0]),
        ctypes.c_double(pos[4, 1]),
        ctypes.c_double(pos[4, 2]),

        ctypes.c_double(pos[5, 0]),
        ctypes.c_double(pos[5, 1]),
        ctypes.c_double(pos[5, 2]),

        ctypes.c_double(pos[6, 0]),
        ctypes.c_double(pos[6, 1]),
        ctypes.c_double(pos[6, 2]))
