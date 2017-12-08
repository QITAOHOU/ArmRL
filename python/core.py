import numpy as np
from numpy.matlib import repmat
import math
import random

class JointProcessor:
  def __init__(self, space=None):
    self.space = space

  def process_env_action(self, discrete):
    """
    If the discrete is the size of the discrete space, then transform it to
    the continuous space
    """
    discrete = np.array(discrete, dtype=np.float32)
    if discrete.shape[0] == self.space.bins.shape[0]:
      print("Already is continuous space:", discrete)
      return discrete
    continuous = np.zeros([self.space.bins.shape[0]], dtype=np.float32)
    idx = 0
    for i in range(self.space.bins.shape[0]):
      for j in range(self.space.bins[i]):
        if discrete[idx + j] > 0:
          continuous[i] = j * self.space.intervals[i] + \
              self.space.ranges[i, 0]
          break
      idx += self.space.bins[i]
    return continuous

  def process_ml_action(self, continuous):
    """
    Otherwise if the continuous is the size of the continuous space, then
    transform it to the discrete space
    """
    continuous = np.array(continuous, dtype=np.float32)
    if continuous.shape[0] == self.space.n:
      print("Already in discrete space:", continuous)
      return continuous
    discrete = np.zeros([self.space.n], dtype=np.float32)
    offsets = np.cumsum(np.concatenate([np.array([0]), self.space.bins[:-1]]))
    for i in range(continuous.shape[0]):
      if continuous[i] < self.space.ranges[i, 0]:
        continuous[i] = self.space.ranges[i, 0]
      elif continuous[i] > self.space.ranges[i, 1]:
        continuous[i] = self.space.ranges[i, 1]
      binidx = int(round((continuous[i] - self.space.ranges[i, 0]) /
        self.space.intervals[i]))
      if binidx < 0:
        binidx = 0
      elif binidx >= self.space.bins[i] - 1:
        binidx = self.space.bins[i] - 1
      discrete[binidx + offsets[i]] = 1.0
    return discrete

  def process_Q(self, dataset):
    qstates = np.concatenate([dataset["states"], dataset["actions"]], axis=1)
    qvalues = dataset["values"]
    return { "data": qstates, "label": qvalues }

#class DiscreteSpace(gym.Space):
class DiscreteSpace:
  def __init__(self, ranges=[], intervals=[], batch_size=32):
    ranges = np.array(ranges, dtype=np.float32)
    intervals = np.array(intervals, dtype=np.float32)
    assert ranges.shape[0] == intervals.shape[0]
    self.ranges = ranges
    self.intervals = intervals
    self.bins = np.array([int(math.floor(ranges[i, 1] - ranges[i, 0]) /
      intervals[i]) + 1 for i in range(ranges.shape[0])], dtype=np.int)
    self.n = np.sum(self.bins)

    self.iter = 0
    self.batch_size = batch_size

  def sample(self, N=1):
    sampleset = np.zeros([N, self.n])
    offsets = np.cumsum(np.concatenate([np.array([0]), self.bins[:-1]]))
    sampleid = 0
    while sampleid < N:
      idx = [random.randint(0, self.bins[i] - 1) + offsets[i]
          for i in range(self.ranges.shape[0])]
      sampleset[sampleid, idx] = 1.0
      if np.sum(np.all(sampleset == np.array([sampleset[sampleid, :]]), \
          axis=1)) == 1:
        sampleid += 1
    return sampleset

  def sampleAll(self):
    idx = np.array(range(np.prod(self.bins)))
    digits = np.cumprod(self.bins)
    shift = np.concatenate([[1], digits[:-1]])
    offset = np.cumsum(np.concatenate([[0], self.bins[:-1]]))
    idx = np.floor_divide(
        np.mod(repmat(np.array([idx]).T, 1, self.bins.shape[0]),
          repmat(np.array([digits]), idx.shape[0], 1)),
        repmat(np.array([shift]), idx.shape[0], 1)) + \
            repmat(np.array([offset]), idx.shape[0], 1)
    sampleset = np.zeros([idx.shape[0], self.n])
    for i in range(idx.shape[0]):
      sampleset[i, idx[i, :]] = 1
    return sampleset

  def __iter__(self):
    self.iter = 0
    return self

  def __next__(self):
    batch_id = self.iter
    batch_size = self.batch_size
    N = np.prod(self.bins)
    if batch_id * batch_size >= N:
      raise StopIteration
    self.iter += 1

    idx = np.array(range(batch_id * batch_size,
      min((batch_id + 1) * batch_size, N)))
    digits = np.cumprod(self.bins)
    shift = np.concatenate([[1], digits[:-1]])
    offset = np.cumsum(np.concatenate([[0], self.bins[:-1]]))
    idx = np.floor_divide(
        np.mod(repmat(np.array([idx]).T, 1, self.bins.shape[0]),
          repmat(np.array([digits]), idx.shape[0], 1)),
        repmat(np.array([shift]), idx.shape[0], 1)) + \
            repmat(np.array([offset]), idx.shape[0], 1)
    sampleset = np.zeros([idx.shape[0], self.n])
    for i in range(idx.shape[0]):
      sampleset[i, idx[i, :]] = 1
    return sampleset

  def contains(self, x):
    return x.shape[0] == self.bins.shape[0] and \
        sum([self.ranges[i, 0] <= x[i] and x[i] <= self.ranges[i, 1]
          for i in range(self.ranges.shape[0])]) == self.ranges.shape[0]

#class ContinuousSpace(gym.Space)
class ContinuousSpace:
  def __init__(self, ranges=[]):
    ranges = np.array(ranges, dtype=np.float32)
    self.ranges = ranges
    self.intervals = np.array([ranges[i, 1] - ranges[i, 0]
      for i in range(ranges.shape[0])], dtype=np.float32)
    self.n = self.ranges.shape[0]

  def sample(self, N=1):
    # hopefully doesn't need fixing
    return np.array([[self.intervals[i] * random.random() + self.ranges[i, 0]
        for i in range(self.ranges.shape[0])] for sampleid in range(N)])

  def contains(self, x):
    return x.shape[0] == self.n and \
        sum([x[i] >= self.ranges[i][0] and x[i] <= self.ranges[i, 1]
          for i in range(self.ranges.shape[0])]) == self.ranges.shape[0]
