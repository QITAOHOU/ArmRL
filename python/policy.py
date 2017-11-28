import random

class EpsilonGreedy:
  def __init__(self, epsilon=0.1, base=10, policyFn=random.random):
    self.epsilon = epsilon
    self.base = base
    self.policyFn = policyFn

  def __call__(self, state):
    if random.random() < self.epsilon:
      return random.randint(0, self.base)
    else:
      return self.policyFn(state)
