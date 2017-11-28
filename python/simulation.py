import ctypes, os
import numpy as np

lib_dirs = os.getenv("GAZEBO_PLUGIN_PATH").split(":")
lib_name = ""
for dir_name in lib_dirs:
  if os.path.isfile(dir_name + "/libarm_plugin.so"):
    lib_name = dir_name + "/libarm_plugin.so"
    break
lib = ctypes.cdll.LoadLibrary(lib_name)
lib.arm_plugin_init.resType = None
lib.arm_plugin_destroy.resType = None
lib.arm_plugin_setPositions.resType = None
lib.arm_plugin_setPositions.argTypes = [ \
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
    self.num_joints = 7

    self.default_length = [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]

    # open the gazebo simulation
    lib.arm_plugin_init()

  def __del__(self):
    lib.arm_plugin_destroy()

  def setPositions(self, pos):
    lib.arm_plugin_setPositions(
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
