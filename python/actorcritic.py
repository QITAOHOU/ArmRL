#!/usr/bin/env python3
import numpy as np
import argparse
import random
import time
import os
import signal
from environment import BasketballVelocityEnv
from core import ContinuousSpace
from models import ActorCritic

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
  parser.add_argument("--load_actor_params", type=str,
      help="Load previously learned actor parameters from [LOAD_PARAMS]")
  parser.add_argument("--load_critic_params", type=str,
      help="Load previously learned critic parameters from [LOAD_PARAMS]")
  parser.add_argument("--save_actor_params", type=str,
      help="Save learned actor parameters to [SAVE_PARAMS]")
  parser.add_argument("--save_critic_params", type=str,
      help="Save learned critic parameters to [SAVE_PARAMS]")
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
  actorFn = MxFullyConnected(sizes=[stateSpace.n + [1], actionSpace.n],
      use_gpu=True)
  criticFn = MxFullyConnected(sizes=[stateSpace.n + actionSpace.n, 1],
      use_gpu=True))
  if args.load_actor_params:
    print("loading actor params...")
    actorFn.load_params(args.load_actor_params)
  if args.load_critic_params:
    print("loading critic params...")
    criticFn.load_params(args.load_critic_params)

  policyFn = EpsilonGreedyPolicy(epsilon=0.1,
      getActionsFn = lambda state: actorFn.predict({
        "data": state,
        "value": 
  if args.logfile:
    log = open(args.logfile, "a")
  
  rollout = 0
  while args.num_rollouts == -1 or rollout < args.num_rollouts:
    print("Iteration:", rollout)
    state = env.reset()
    reward = 0
    done = False
    steps = 0
    value = np.array([0])
    while not done:
      if stopsig:
        break
      action = actorFn(np.concatenate([state, value]))
      nextState, reward, done, info = env.step(action)
      value = criticFn(np.concatenate([state, action]))
      dataset.append(state, action, nextState, reward)
      state = nextState
      steps += 1
      if args.render and rollout % args.render_interval == 0:
        env.render()
    if stopsig:
      break

    actorFn.fit()
    actorFn.clear()
    print("Reward:", reward if (reward >= 0.00001) else 0, "with Error:",
        actorFn.score(), "with steps:", steps)
    if args.logfile:
      log.write("[" + str(rollout) + ", " + str(reward) + "]\n")
    rollout += 1

  if args.logfile:
    log.close()
  if args.save_actor_params:
    print("saving actor params...")
    actorFn.save_params(args.save_actor_params)
  if args.save_critic_params:
    print("saving critic params...")
    criticFn.save_params(args.save_critic_params)

if __name__ == "__main__":
  main()
