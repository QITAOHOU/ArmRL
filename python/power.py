#!/usr/bin/env python3
import numpy as np
import argparse
import random
import time
import os
import signal
from environment import BasketballVelocityEnv
from core import ContinuousSpace
from models import PoWERDistribution

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
  parser.add_argument("--logfile", type=str,
      help="Indicate where to save rollout data")
  parser.add_argument("--load_params", type=str,
      help="Load previously learned parameters from [LOAD_PARAMS]")
  parser.add_argument("--save_params", type=str,
      help="Save learned parameters to [SAVE_PARAMS]")
  args = parser.parse_args()

  signal.signal(signal.SIGINT, stopsigCallback)
  global stopsig

  # create the basketball environment
  env = BasketballVelocityEnv(fps=60.0, timeInterval=0.1,
      goal=[0, 5, 0],
      initialLengths=np.array([0, 0, 1, 1, 0, 1, 1]),
      initialAngles=np.array([-5, 45, -10, -10, -5, -10, -5]))

  # create space
  stateSpace = ContinuousSpace(ranges=env.state_range())
  actionSpace = ContinuousSpace(ranges=env.action_range())

  # create the model and policy functions
  modelFn = PoWERDistribution(stateSpace.n, actionSpace.n)
  if args.load_params:
    print("loading params...")
    modelFn.load_params(args.load_params)
  if args.logfile:
    log = open(args.logfile, "a")
  
  rollout = 0
  while args.num_rollouts == -1 or rollout < args.num_rollouts:
    print("Iteration:", rollout)
    state = env.reset()
    reward = 0
    done = False
    steps = 0
    while not done:
      if stopsig:
        break
      action = modelFn.predict(state)
      nextState, reward, done, info = env.step(action)
      modelFn.append(state, action, nextState, reward)
      state = nextState
      steps += 1
      if args.render and rollout % args.render_interval == 0:
        env.render()
    if stopsig:
      break

    # no importance sampling just yet, do it later
    modelFn.fit()
    modelFn.clear()
    print("Reward:", reward if (reward >= 0.00001) else 0, "with Error:",
        modelFn.score(), "with steps:", steps)
    if args.logfile:
      log.write("[" + str(rollout) + ", " + str(reward) + "]\n")
    rollout += 1

  if args.logfile:
    log.close()
  if args.save_params:
    print("saving params...")
    modelFn.save_params(args.save_params)

if __name__ == "__main__":
  main()
