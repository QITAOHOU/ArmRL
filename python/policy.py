import random

class EpsilonGreedyPolicy:
  def __init__(self, epsilon=0.1, base=10, policyFn=None, randomFn=None):
    self.epsilon = epsilon
    self.base = base
    self.policyFn = policyFn
    self.randomFn = randomFn

  def __call__(self, state):
    if random.random() < self.epsilon:
      if self.randomFn:
        return self.randomFn()
      else:
        return None
    else:
      if self.policyFn:
        return self.policyFn(state)
      else:
        return None
