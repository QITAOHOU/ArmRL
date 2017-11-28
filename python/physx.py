import numpy as np
from math import sin, cos, radians

def RotZ(theta):
  c = cos(theta)
  s = sin(theta)
  return np.array([
    [c, -s, 0, 0],
    [s, c, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]).astype(np.float32)

def RotY(theta):
  c = cos(theta)
  s = sin(theta)
  return np.array([
    [c, 0, s, 0],
    [0, 1, 0, 0],
    [-s, 0, c, 0],
    [0, 0, 0, 1]]).astype(np.float32)

def RotX(theta):
  c = cos(theta)
  s = sin(theta)
  return np.array([
    [1, 0, 0, 0],
    [0, c, -s, 0],
    [0, s, c, 0],
    [0, 0, 0, 1]]).astype(np.float32)

def Translate(x, y, z):
  return np.array([
    [1, 0, 0, x],
    [0, 1, 0, y],
    [0, 0, 1, z],
    [0, 0, 0, 1]]).astype(np.float32)

def forwardKinematics(lengths, angles):
  """
  Given a series of lengths and angles, calculate the endpoints with the
  following joint configurations:
    angle 1: z
    angle 2: x
    angle 3: x
    angle 4: x
    angle 5: z
    angle 6: x
    angle 7: doesn't matter
  Arguments:
  - lengths: the lengths of each of the links
  - angles: the angles of each of the joints in degrees
  Return:
  - a 7x3 numpy array containing all the positions
  """
  # convert all the angles into radians
  angles = [radians(theta) for theta in angles]
  transformations = [
      lambda v: np.dot(np.dot(Translate(0, 0, lengths[0]), RotZ(angles[0])), v),
      lambda v: np.dot(np.dot(Translate(0, 0, lengths[1]), RotX(angles[1])), v),
      lambda v: np.dot(np.dot(Translate(0, 0, lengths[2]), RotX(angles[2])), v),
      lambda v: np.dot(np.dot(Translate(0, 0, lengths[3]), RotX(angles[3])), v),
      lambda v: np.dot(np.dot(Translate(0, 0, lengths[4]), RotZ(angles[4])), v),
      lambda v: np.dot(np.dot(Translate(0, 0, lengths[5]), RotX(angles[5])), v),
      lambda v: np.dot(np.dot(Translate(0, 0, lengths[6]), RotY(angles[6])), v)]
  positions = []
  for i in range(0, len(transformations)):
    positions = [np.array([0, 0, 0, 1])] + positions
    ind = len(transformations) - i - 1
    for j, pos in enumerate(positions):
      positions[j] = transformations[ind](pos)
  return np.array(positions)[:,:3]
