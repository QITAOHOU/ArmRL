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
      dist = self.distribution(np.concatenate([repmat(state, len(actions), 1),
        np.array(actions)], axis=1))
      dist /= np.sum(dist) # normalize
      dist = np.cumsum(dist)
      dist[-1] = 1.0
      p = random.random()
      for i in range(len(dist)):
        if p <= dist[i]:
          return actions[i]
      return actions[-1]
    else:
      return actions[random.randint(0, len(actions) - 1)]

class EpsilonGreedyPolicy(BasePolicy):
  def __init__(self, epsilon=0.1, getActionsFn=None, distributionFn=None):
    super().__init__(getActionsFn, distributionFn)
    self.epsilon = epsilon

  def __call__(self, state):
    actions = self.getActions(state)
    if self.distribution == None or random.random() < self.epsilon:
      return actions[random.randint(0, len(actions) - 1)]
    else:
      dist = self.distribution(np.concatenate([repmat(state, len(actions), 1),
        np.array(actions)], axis=1))[:actions.shape[0], :]
      return actions[np.argmax(dist)]
