import numpy as np
import simulation
import physx
import time
import math
#from gym import Env, spaces

#class BasketballEnv(gym.Env):
class BasketballEnv:
  """
  State: [7 joint angular positions + release,
          7 joint angular velocities + release]
  Action: [7 joint angular velocities + release]
  """
  def __init__(self, goal=np.array([5, 0, 0]), \
      initialAngles=np.zeros([7], dtype=np.float32), \
      initialLengths=np.array([0, 0, 1, 1, 1, 0, 1], dtype=np.float32),
      fps=30.0):
    self.initial_angles = initialAngles.astype(np.float32)
    self.lengths = initialLengths.astype(np.float32)
    self.angles = self.initial_angles.copy()
    self.sim = None
    self.time_render = None
    self.fps = fps
    self.iter = 0

    self.goal = goal

  def reset(self):
    self.iter = 0
    self.angles = self.initial_angles.copy()
    self.time_render = None
    return np.concatenate([self.angles, np.zeros(9)])

  def step(self, action):
    action = action.astype(np.float32)
    # take the 0.1 position interval
    self.angles += 1.0 / self.fps * action[:-1]
    nextState = np.concatenate([self.angles, [action[7]], action])
    reward = self.rewardFn(self.angles, action)
    done = self.terminationFn(self.angles, action)
    self.iter += 1
    # assuming that actions are immediately applied to become the velocities
    return nextState, reward, done, {}

  def render(self):
    pos = physx.forwardKinematics(self.lengths, self.angles)
    if self.sim == None:
      self.sim = simulation.Arm()
      self.sim.default_length = self.lengths
    self.sim.setPositions(pos)
    t = time.time()
    if self.time_render == None or self.time_render < t:
      self.time_render = t + 1.0 / self.fps
    else:
      time.sleep(self.time_render - t)
      self.time_render += 1.0 / self.fps

  def rewardFn(self, state, action):
    if action[7] < 0.5:
      return 0.0

    # compute the final position and velocity
    joints = action[:-1]
    pos = physx.forwardKinematics(self.lengths, self.angles)[-1, :]
    p0 = physx.forwardKinematics(self.lengths, self.angles - 0.005 * joints)
    p1 = physx.forwardKinematics(self.lengths, self.angles + 0.005 * joints)
    vel = (p1[-1,:3] - p0[-1,:3]) / 0.01

    # compute the time it would take to reach the goal (kinematics equations)
    g = -4.9
    vz = vel[2]
    dz = pos[2] - self.goal[2]
    # quadratic formula (+/- sqrt)
    # the parabola doesn't even reach the line
    b2_4ac = vz * vz - 4 * g * dz
    if b2_4ac < 0:
      return 0.0
    dt1 = (-vz + math.sqrt(b2_4ac)) / (2 * g)
    dt2 = (-vz - math.sqrt(b2_4ac)) / (2 * g)
    dt = max(dt1, dt2)
    # the ball would have to go backwards in time to reach the goal
    if dt < 0:
      return 0.0

    # find the distance from the goal (in the xy-plane) that the ball has hit
    print("Ball landed at:", pos[:2] + vel[:2] * dt)
    dp = self.goal[:2] - (pos[:2] + vel[:2] * dt)
    # use a kernel distance
    return np.exp(-np.dot(dp, dp))

  def terminationFn(self, state, action):
    return action[7] >= 0.5 or self.iter >= 1000

  def state_range(self):
    return [(-180.0, 180.0),  # positions
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (0, 1),
            (-180.0, 180.0),  # actions
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (0.0, 1.0)]

  def action_range(self):
    return [(-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (0.0, 1.0)]
