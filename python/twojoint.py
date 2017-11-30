#!/usr/bin/env python3
import numpy as np
import argparse
import random
import time
import os
import signal
from environment import BasketballEnv
from core import ContinuousSpace, \
                 DiscreteSpace, \
                 JointProcessor
from models import MxFullyConnected
from policy import EpsilonGreedyPolicy
from memory import Trajectory

def createAction(joint1, joint2, release):
  return np.array([0, joint1, joint2, 0, 0, 0, 0, release],
      dtype=np.float32)

def randomSample():
  return np.array([
    random.randint(-36, 36) * 5,
    random.randint(-36, 36) * 5,
    (random.random() > 0.90)
    ], dtype=np.int)

class GreedyPolicy:
  def __init__(self, actionSpace, model):
    self.actionSpace = actionSpace
    self.model = model

  def __call__(self, state):
    validActions = []
    while len(validActions) < 32:
      a = self.actionSpace.sample()
      resample = False
      for action in validActions:
        if np.array_equal(a, action):
          resample = True
          break
      if not resample:
        validActions.append(a)
    qstates = np.array([np.concatenate([state, action])
      for action in validActions])
    qvalues = self.model(qstates)
    qvalues = np.concatenate(qvalues)
    # sample an action from the policy (use softmax)
    e_x = np.exp(qvalues)
    pi = e_x / np.sum(e_x)
    pi = np.cumsum(pi)
    pi[-1] = 1.0
    idx = random.random()
    for i in range(len(pi)):
      if idx <= pi[i]:
        idx = i
        break
    return validActions[idx]

stopsig = False
def stopsigCallback(signo, idx):
  global stopsig
  stopsig = True

def main():
  signal.signal(signal.SIGINT, stopsigCallback)
  global stopsig

  # define arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--render", action="store_true",
      help="Render the state")
  args = parser.parse_args()

  # create the basketball environment
  env = BasketballEnv(fps=60.0,
      goal=[0, 5, 0],
      initialLengths=np.array([0, 0, 1, 1, 0, 0, 0]),
      initialAngles=np.array([0, 45, 0, 0, 0, 0, 0]))

  # create which space we want for the states and actions
  stateSpace = ContinuousSpace(ranges=env.state_range())
  actionRange = env.action_range()
  actionSpace = DiscreteSpace(intervals=[5 for i in range(2)] + [1],
      ranges=[actionRange[1], actionRange[2], actionRange[7]])

  # create a transformation processor between the env and the space
  processor = JointProcessor(actionSpace)

  # create the model and policy functions
  modelFn = MxFullyConnected(sizes=[stateSpace.n + actionSpace.n, 64, 64, 1],
      use_gpu=True)
  if "model_params" in os.listdir("."):
    print("loading params...")
    modelFn.model.load_params("model_params")
  agentPolicy = GreedyPolicy(actionSpace, modelFn)
  policyFn = EpsilonGreedyPolicy(
      policyFn=agentPolicy,
      randomFn=lambda: processor.process_ml_action(randomSample()))
  dataset = Trajectory(0.8)

  for i in range(1000):
    if stopsig:
      break
    print("Iteration", i)
    state = env.reset()
    reward = 0
    done = False
    while not done:
      if stopsig:
        break
      action = policyFn(state)
      A = processor.process_env_action(action)
      A = createAction(A[0], A[1], A[2])
      nextState, reward, done, info = env.step(A)
      dataset.append(state, action, reward)
      state = nextState
      if args.render and i % 10 == 0:
        env.render()
      if i % 100 == 0:
        policyFn.epsilon = 0.1 - (float(i) / 10000)
        print("Epsilon is now:", policyFn.epsilon)
    if stopsig:
      break
    if reward < 0.000001:
      R = 0
    else:
      R = reward
    print("Reward:", R)
    dataset.reset()
    data = dataset.sample()
    modelFn.fit({
      "qstates": np.concatenate([data["states"], data["actions"]], axis=1),
      "qvalues": data["values"]
      }, num_epochs=10)
  print("saving params...")
  modelFn.model.save_params("model_params")

if __name__ == "__main__":
  main()
