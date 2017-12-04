import numpy as np
import simulation
import physx
import time
import math
#from gym import Env, EnvSpec

class EnvSpec:
  def __init__(self, timestep_limit=1000):
    self.max_episode_steps = timestep_limit

  @property
  def timestep_limit(self):
    return self.max_episode_steps

# TODO: make sure the state gets stored as a 1-d vector

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
      timeInterval=1.0, fps=30.0):
    self.lengths = initialLengths.astype(np.float32)
    self.initial_state = np.concatenate([initialAngles.astype(np.float32),
      np.zeros([8], dtype=np.float32)]) # angpos, angvel, release
    self.angles = self.initial_state.copy()
    self.goal = goal
    self.sim = None
    self.time_render = None
    self.fps = fps
    self.iter = 0
    self.spec = EnvSpec()
    self.time_interval = timeInterval

  def reset(self):
    self.iter = 0
    self.angles = self.initial_state.copy()
    self.time_render = None
    return self.angles.copy()

  def step(self, action):
    actionRange = np.array(self.action_range(), dtype=np.float32)
    action = action.astype(np.float32)
    action = np.minimum(np.maximum(action, actionRange[:,0]), actionRange[:,1])
    t = self.time_interval
    t2 = 0.5 * t ** 2.0
    A = np.concatenate([
      np.concatenate([np.eye(7), np.eye(7) * t, np.zeros([7, 1])], axis=1),
      np.concatenate([np.zeros([7, 7]), np.eye(7), np.zeros([7, 1])], axis=1),
      np.concatenate([np.zeros([1, 14]), np.array([[1.0]])], axis=1)
      ])
    B = np.concatenate([
      np.concatenate([np.eye(7) * t2, np.zeros([7, 8])], axis=1),
      np.concatenate([
        np.zeros([7, 7]), np.eye(7) * t, np.zeros([7, 1])], axis=1),
      np.concatenate([np.zeros([1, 14]), np.array([[1.0]])], axis=1)
      ])
    u = np.concatenate([
      np.array([action[:-1]]).T,
      np.array([action[:-1]]).T,
      np.array([[action[-1]]])
      ])
    # solve for next position/velocity, along with limits
    nextState = ((np.dot(A, np.array([self.angles]).T) + np.dot(B, u)).T)[0]
    stateRange = np.array(self.state_range(), dtype=np.float32)
    minState = np.maximum(stateRange[:,0],
        np.concatenate([stateRange[:7,0] + self.angles[:7], self.angles[7:]]))
    maxState = np.minimum(stateRange[:,1],
        np.concatenate([stateRange[:7,1] + self.angles[:7], self.angles[7:]]))
    nextState = np.maximum(minState, nextState)
    nextState = np.minimum(maxState, nextState)
    # find reward and termination
    reward = self.rewardFn(nextState, action) # should be previous state
    done = self.terminationFn(nextState, action)
    self.angles = nextState
    self.iter += 1
    # assuming that actions are immediately applied to become the velocities
    return nextState, reward, done, {}

  def render(self, mode="human", close=False):
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

  def close(self):
    if self.sim != None:
      del self.sim
      self.sim = None

  def seed(self, seed=None):
    pass

  def rewardFn(self, state, action):
    if state[-1] < 0.5:
      return 0.0

    # compute the final position and velocity
    joints = states[7:-1]
    pos = physx.forwardKinematics(self.lengths, state[:7])[-1, :]
    p0 = physx.forwardKinematics(self.lengths, state[:7] - 0.005 * joints)
    p1 = physx.forwardKinematics(self.lengths, state[:7] + 0.005 * joints)
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
    dp = self.goal[:2] - (pos[:2] + vel[:2] * dt)
    # use a kernel distance
    return np.exp(-np.dot(dp, dp))

  def terminationFn(self, state, action):
    return action[7] >= 0.5 or self.iter >= self.spec.timestep_limit

  def state_range(self):
    return [(-180.0, 180.0),  # positions
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),  # velocities (max RPM)
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (-180.0, 180.0),
            (0.0, 1.0)]       # release

  def action_range(self):
    return [(-20.0, 20.0),
            (-20.0, 20.0),
            (-20.0, 20.0),
            (-20.0, 20.0),
            (-20.0, 20.0),
            (-20.0, 20.0),
            (-20.0, 20.0),
            (0.0, 1.0)]

  def __del__(self):
    self.close()
