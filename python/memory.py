import numpy as np
import random
from copy import deepcopy

MAX_LIMIT = 10000

# TODO: fixme

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

class ReplayBuffer:
  def __init__(self):
    self.T = []
    self.D = []

  def append(self, state, action, reward, nextState=None):
    self.T.append({
      "state": state,
      "action": action,
      "nextState": nextState,
      "reward": reward
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
    nextActions = []
    values = []
    for i in range(min(num_items, len(idx))):
      states.append(dataset[idx[i]]["state"])
      actions.append(dataset[idx[i]]["action"])
      rewards.append(dataset[idx[i]]["reward"])
      nextStates.append(dataset[idx[i]]["nextState"])
      # do a hack to speed up the training
      if idx[i] + 1 in ends:
        nextActions.append(np.zeros(dataset[0]["action"].shape,
          dtype=np.float32)) # usually this is the "best action"
      else:
        nextActions.append(dataset[idx[i] + 1]["action"])
      # careful: wrong place to put it, but better than diverging tbh
      values.append(dataset[idx[i]]["value"])
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "nextStates": np.array(nextStates, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "nextActions": np.array(nextActions, dtype=np.float32),
        "values": np.array(values, dtype=np.float32)
        }

  def sampleLast(self):
    states = []
    actions = []
    nextStates = []
    rewards = []
    values = []
    for i in range(len(self.D[-1])):
      states.append(self.D[-1][i]["state"])
      actions.append(self.D[-1][i]["action"])
      rewards.append(self.D[-1][i]["rewards"])
      nextStates.append(dataset[idx[i]]["nextState"])
      # careful: wrong place to put it, but better than diverging tbh
      values.append(dataset[idx[i]]["value"])
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "nextStates": np.array(nextStates, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "values": np.array(values, dtype=np.float32)
        }

  def clear(self):
    self.D = []
