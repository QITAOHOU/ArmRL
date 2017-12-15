import numpy as np
from numpy.matlib import repmat
import random

class BasePolicy: # sample-based policy
  def __init__(self, getActionsFn, distributionFn=None):
    self.getActions = getActionsFn
    self.distribution = distributionFn
    self.chosenAction = None
    self.actions = None

  def __call__(self, state):
    if self.getActions == None:
      return np.array([])
    self.actions = self.getActions(state)
    if type(self.actions) == type(np.array([])):
      self.actions = list(self.actions)
    # default behavior is to return a random action sampled uniformly
    # otherwise we sample
    if self.distribution:
      dist = self.distribution(np.concatenate([
        repmat(state, len(self.actions), 1), np.array(self.actions)], axis=1))
      self.chosenAction = np.random.choice(dist.shape[0], p=dist)
    else:
      self.chosenAction = np.random.choice(len(self.actions))
    return self.actions[self.chosenAction]

class EpsilonGreedyPolicy(BasePolicy):
  def __init__(self, epsilon=0.1, getActionsFn=None, distributionFn=None,
      randomFn=None, processor=None):
    super().__init__(getActionsFn, distributionFn)
    self.randomFn = randomFn
    if self.randomFn == None:
      self.randomFn = BasePolicy(getActionsFn)
    self.epsilon = epsilon
    self.processor = processor

  def __call__(self, state):
    if self.getActions == None:
      return np.array([])
    self.actions = self.getActions(state)
    if type(self.actions) == type(np.array([])):
      self.actions = list(self.actions)
    if self.distribution and random.random() >= self.epsilon:
      if self.processor:
        qstate = self.processor.process_Qstate(
            repmat(state, len(self.actions), 1), self.actions)
      else:
        qstate = np.concatenate([repmat(state, len(self.actions), 1),
          np.array(self.actions)], axis=1)
      self.chosenAction = np.argmax(self.distribution(qstate))
    else:
      self.chosenAction = np.random.choice(len(self.actions))
    return self.actions[self.chosenAction]
