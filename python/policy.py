import numpy as np
from numpy.matlib import repmat
import random

class BasePolicy: # sample-based policy
  def __init__(self, getActionsFn, distributionFn=None):
    self.getActions = getActionsFn
    self.distribution = distributionFn

  def __call__(self, state):
    if self.getActions == None:
      return np.array([])
    actions = self.getActions(state)
    if type(actions) == type(np.array([])):
      actions = list(actions)
    # default behavior is to return a random action sampled uniformly
    # otherwise we sample
    if self.distribution:
      dist = self.distribution(np.concatenate([
        repmat(state, len(actions), 1), np.array(actions)], axis=1))
      return actions[np.random.choice(dist.shape[0], p=dist)]
    else:
      return actions[np.random.choice(len(actions))]

class EpsilonGreedyPolicy(BasePolicy):
  def __init__(self, epsilon=0.1, getActionsFn=None, distributionFn=None,
      randomFn=None):
    super().__init__(getActionsFn, distributionFn)
    self.randomFn = randomFn
    if self.randomFn == None:
      self.randomFn = BasePolicy(getActionsFn)
    self.epsilon = epsilon

  def __call__(self, state):
    if self.getActions == None:
      return np.array([])
    actions = self.getActions(state)
    if type(actions) == type(np.array([])):
      actions = list(actions)
    if self.distribution and random.random() >= self.epsilon:
      dist = self.distribution(np.concatenate([
        repmat(state, len(actions), 1), np.array(actions)], axis=1))
      return actions[np.argmax(dist)]
    else:
      return actions[np.random.choice(len(actions))]
