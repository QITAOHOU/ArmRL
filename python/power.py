#!/usr/bin/env python3
import numpy as np
import argparse
import random
import time
import os
import signal
from environment import BasketballVelocityEnv
from core import ContinuousSpace
from models import MxFullyConnected

stopsig = False
def stopsigCallback(signo, _):
  global stopsig
  stopsig = True

def main():
  # define arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--render", action="store_true",
      help="Render the state")
  parser.add_argument("--render_interval", type=int, default=10,
      help="Number of rollouts to skip before rendering")
  parser.add_argument("--num_rollouts", type=int, default=-1,
      help="Number of max rollouts")
  args = parser.parse_args()

  # create space
  stateSpace = ContinuousSpace(ranges=env.state_range())
  actionSpace = ContinuousSpace(ranges=env.action_range())

  # create the model and policy functions
  modelFn = MxFullyConnected(sizes=[stateSpace.n, actionSpace.n], alpha=0.001,
      use_gpu=True)
  dataset = ReplayBuffer()
  
  rollout = 0
  while args.num_rollouts == -1 or rollout < args.num_rollouts:
    state = env.reset()
    reward = 0
    done = False
    steps = 0
    while not done:
      if stopsig:
        break
      # sample according to action distribution
      action = actionSpace.sample(modelFn(state))
      nextState, reward, done, info = env.step(action)
      dataset.append(state, action, reward, nextState)
      state = nextState
      steps += 1
    if stopsig:
      break

    # no importance sampling just yet
    dataset.reset()
    modelFn.fit(dataset.sampleLast(), num_epochs=10)
    dataset.clear()
    print("Reward:", reward if (reward >= 0.00001) else 0)

    rollout += 1

if __name__ == "__main__":
  main()
