import numpy as np

class Trajectory:
  def __init__(self, gamma):
    self.gamma = gamma
    self.T = []
    self.D = []

  def append(self, state, action, reward):
    self.T.append({"state": state, "action": action, "reward": reward})
  
  def reset(self):
    self.T[-1]["value"] = self.T[-1]["reward"]
    for i in range(len(self.T) - 1):
      s_ = len(self.T) - i - 1
      s = s_ - 1
      self.T[s]["value"] = self.T[s]["reward"] + \
          self.gamma * self.T[s_]["value"]

    self.D.append(self.T)
    self.T = []

  def sample(self, num_items=-1):
    if len(self.T) < num_items:
      if len(self.D) > 0 and len(self.T) + len(self.D[-1]) < num_items:
        return []

    dataset = []
    for d in self.D:
      dataset += d
    idx = list(range(0, len(dataset)))
    random.shuffle(idx)

    sampled = []
    if num_items == -1:
      sampled = self.T
      batch_size = 32
      num_items = batch_size
      while len(sampled) % batch_size > 0 and len(sampled) < num_items:
        sampled.append(dataset[idx[i]])
    else:
      for i in range(num_items):
        sampled.append(dataset[idx[i]])
    return sampled
