import numpy as np
import random

MAX_LIMIT = 10000

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

    global MAX_LIMIT
    while sum([len(d) for d in self.D]) > MAX_LIMIT:
      self.D = self.D[1:]

  def sample(self, num_items=-1):
    dataset = []
    for d in self.D:
      dataset += d
    idx = list(range(0, len(dataset)))
    random.shuffle(idx)

    if num_items == -1:
      num_items = len(idx)

    states = []
    actions = []
    values = []
    for i in range(min(num_items, len(idx))):
      states.append(dataset[idx[i]]["state"])
      actions.append(dataset[idx[i]]["action"])
      values.append(dataset[idx[i]]["value"])
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "values": np.array(values, dtype=np.float32)
        }

  def sampleLast(self):
    states = []
    actions = []
    values = []
    for i in range(len(self.D[-1])):
      states.append(self.D[-1][i]["state"])
      actions.append(self.D[-1][i]["action"])
      values.append(self.D[-1][i]["value"])
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "values": np.array(values, dtype=np.float32)
        }

  def clear(self):
    self.D = []
