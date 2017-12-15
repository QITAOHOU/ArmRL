import numpy as np
import random
from copy import deepcopy

MAX_LIMIT = 4096

def Bellman(d, gamma):
  d = deepcopy(d)
  values = []
  rewards = [x["reward"] for x in d]
  for i, r in enumerate(rewards[::-1]):
    if i == 0:
      values.append(r)
    else:
      values.append(values[i-1] * gamma + r)
  values = values[::-1]
  for i in range(len(d)):
    d[i]["value"] = values[i]
  return d

class RingBuffer:
  def __init__(self, max_limit=MAX_LIMIT):
    self.D = []
    self.max_limit = max_limit
    self.iter = 0

  def append(self, x):
    if len(self.D) < self.max_limit:
      self.D.append(x)
    else:
      self.D[self.iter] = x
      self.iter = (self.iter + 1) % self.max_limit

  def sample(self, n=-1):
    dataset = []
    idx = list(range(len(self.D)))
    random.shuffle(idx)
    if n == -1:
      n = len(self.D)
    n = min(n, len(self.D))
    for i in range(n):
      dataset.append(self.D[idx[i]])
    return dataset

  def reset(self):
    pass

  def clear(self):
    self.D = []

class ReplayBuffer:
  def __init__(self):
    self.T = []
    self.D = []

  def append(self, state, action, reward, nextState=None, info=None):
    self.T.append({
      "state": state,
      "action": action,
      "nextState": nextState,
      "reward": reward,
      "info": info
      })
  
  def reset(self):
    self.D.append(self.T)
    self.T = []

    global MAX_LIMIT
    while sum([len(d) for d in self.D]) > MAX_LIMIT:
      self.D = self.D[1:]

  def sample(self, num_items=-1, gamma=0.9):
    dataset = []
    ends = []
    for d in self.D:
      dataset += Bellman(d, gamma)
      ends.append(len(dataset))
    idx = list(range(len(dataset)))
    random.shuffle(idx)

    if num_items == -1:
      num_items = len(idx)

    states = []
    actions = []
    nextStates = []
    rewards = []
    terminal = []
    values = []
    info = []
    for i in range(min(num_items, len(idx))):
      states.append(dataset[idx[i]]["state"])
      actions.append(dataset[idx[i]]["action"])
      rewards.append(dataset[idx[i]]["reward"])
      nextStates.append(dataset[idx[i]]["nextState"])
      terminal.append(int(idx[i] + 1 in ends))
      # ===== EXTRA =====
      values.append(dataset[idx[i]]["value"])
      info.append(dataset[idx[i]]["info"])
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "nextStates": np.array(nextStates, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "terminal": np.array(terminal, dtype=np.int),
        "values": np.array(values, dtype=np.float32),
        "info": info
        }

  def sampleLast(self):
    states = []
    actions = []
    nextStates = []
    rewards = []
    values = []
    info = []
    for i in range(len(self.D[-1])):
      states.append(self.D[-1][i]["state"])
      actions.append(self.D[-1][i]["action"])
      rewards.append(self.D[-1][i]["rewards"])
      nextStates.append(dataset[idx[i]]["nextState"])
      # ===== EXTRA =====
      values.append(dataset[idx[i]]["value"])
      info.append(dataset[idx[i]]["info"])
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "nextStates": np.array(nextStates, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "info": info
        }

  def clear(self):
    self.D = []
