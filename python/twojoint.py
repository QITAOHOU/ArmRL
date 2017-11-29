#!/usr/bin/env python3
import numpy as np
import argparse
import random
import time
import os
from environment import BasketballEnv
from core import ContinuousSpace, \
                 DiscreteSpace, \
                 JointProcessor
from models import MxFullyConnected
from policy import EpsilonGreedy

def createAction(joint0, joint1, joint2, release):
  return np.array([joint0, joint1, joint2, 0, 0, 0, 0, release],
      dtype=np.float32)

def randomSample():
  return np.array([
    random.randint(-36, 36) * 5,
    random.randint(-36, 36) * 5,
    random.randint(-36, 36) * 5,
    (random.random() > 0.90)
    ], dtype=np.int)

def main():
  # define arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--render", action="store_true",
      help="Render the state")
  args = parser.parse_args()

  # create the basketball environment
  env = BasketballEnv(fps=60.0,
      initialLengths=np.array([0, 0, 1, 1, 0, 0, 0]),
      initialAngles=np.array([0, 45, -90, 0, 0, 0, 0]))

  # create which space we want for the states and actions
  stateSpace = ContinuousSpace(ranges=env.state_range())
  actionRange = env.action_range()
  actionSpace = DiscreteSpace(intervals=[5 for i in range(3)] + [1],
      ranges=[actionRange[0], actionRange[1], actionRange[2], actionRange[7]])

  # create a transformation processor between the env and the space
  processor = JointProcessor(actionSpace)

  # create the model and policy functions
  qStateSize = stateSpace.n + actionSpace.n
  modelFn = MxFullyConnected(sizes=[stateSpace.n + actionSpace.n, 1])
  staticAction = processor.process_ml_action([0, 30, -30, 0])
  policyFn = EpsilonGreedy(
      policyFn=lambda state: staticAction,
      randomFn=lambda: processor.process_ml_action(randomSample()))

  for i in range(100):
    print("Iteration", i)
    state = env.reset()
    done = False
    while not done:
      action = policyFn(state)
      a = modelFn({
        "qstates": np.concatenate([
          np.array([np.concatenate([state, action])]),
          np.zeros([modelFn.batch_size - 1, qStateSize], dtype=np.float32)])
        })
      action = processor.process_env_action(action)
      action = createAction(action[0], action[1], action[2], action[3])
      nextState, reward, done, info = env.step(action)
      state = nextState
      if args.render:
        env.render()

if __name__ == "__main__":
  main()
