#!/usr/bin/env python3
import numpy as np
import argparse
import time
import os
from environment import BasketballEnv
#from policy import EpsilonGreedy

def createAction(joint1, joint2, release):
  return np.array([0, joint1, joint2, 0, 0, 0, 0, release], dtype=np.float32)

def main():
  # define arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--render", action="store_true",
      help="Render the state")
  args = parser.parse_args()

  # create the basketball environment
  env = BasketballEnv(fps=60.0,
      initialLengths=np.array([0, 0, 1, 1, 0, 0, 0]),
      initialAngles=np.array([0, 45, 90, 0, 0, 0, 0]))
  #model = models.FullyConnected(
  #    sizes=[env.state_size() + env.action_size(), 1])
  #policyFn = EpsilonGreedy(policyFn=model)

  for i in range(100):
    state = env.reset()
    done = False
    for i in range(100):
      #action = policyFn(state)
      action = np.array([1, 2, 3, 4, 5, 6, 1, 0]) * 10
      nextState, reward, done, _ = env.step(action)
      print(nextState, reward, done)
      state = nextState
      if args.render:
        env.render()

if __name__ == "__main__":
  main()
