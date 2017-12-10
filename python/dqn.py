#!/usr/bin/env python3
import numpy as np
from numpy.matlib import repmat
import argparse
import random
import time
import os
import signal
from environment import BasketballAccelerationEnv, BasketballVelocityEnv
from core import ContinuousSpace, \
                 DiscreteSpace, \
                 JointProcessor
from models import MxFullyConnected
from policy import EpsilonGreedyPolicy
from memory import ReplayBuffer

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
  parser.add_argument("--silent", action="store_true",
      help="Suppress print of the DQN config")
  parser.add_argument("--gamma", type=float, default=0.9,
      help="Discount factor")
  parser.add_argument("--epsilon", type=float, default=0.1,
      help="Random factor (for Epsilon-greedy)")
  parser.add_argument("--eps_decay", action="store_true",
      help="Let epsilon decay over time")
  args = parser.parse_args()

  signal.signal(signal.SIGINT, stopsigCallback)
  global stopsig

  # create the basketball environment
  env = BasketballVelocityEnv(fps=60.0, timeInterval=0.1,
      goal=[0, 5, 0],
      initialLengths=np.array([0, 0, 1, 1, 1, 0, 1]),
      initialAngles=np.array([0, 45, -20, -20, 0, 20, 0]))

  # create space
  stateSpace = ContinuousSpace(ranges=env.state_range())
  actionSpace = DiscreteSpace(intervals=[180 for i in range(7)] + [1],
      ranges=env.action_range())
  processor = JointProcessor(actionSpace)

  # create the model and policy functions
  modelFn = MxFullyConnected(sizes=[stateSpace.n + actionSpace.n,
    1024, 1024, 1], alpha=0.001, use_gpu=True)
  if args.load_params:
    print("Loading params...")
    modelFn.load_params(args.load_params)

  softmax = lambda s: np.exp(s) / np.sum(np.exp(s))
  allActions = actionSpace.sampleAll()
  policyFn = EpsilonGreedyPolicy(epsilon=args.epsilon,
      getActionsFn=lambda state: allActions,
      #getActionsFn=lambda state: actionSpace.sample(2048),
      distributionFn=lambda qstate: softmax(modelFn(qstate)))
  dataset = ReplayBuffer()
  if args.logfile:
    log = open(args.logfile, "a")

  if not args.silent:
    print("Env space range:", env.state_range())
    print("Env action range:", env.action_range())
    print("State space:", stateSpace.n)
    print("Action space:", actionSpace.n)
    print("Action space bins:", actionSpace.bins)
    print("Epsilon:", args.epsilon)
    print("Epsilon decay:", args.eps_decay)
    print("Gamma:", args.gamma)
    print("Actions are sampled:",
        policyFn.getActions(None).shape[0] == actionSpace.n)

  rollout = 0
  lastQ = 0
  lastR = 0
  lastSteps = 0
  while args.num_rollouts == -1 or rollout < args.num_rollouts:
    print("Iteration:", rollout)
    state = env.reset()
    reward = 0
    done = False
    steps = 0
    while not done:
      if stopsig:
        break
      action = policyFn(state)
      nextState, reward, done, info = env.step(
          processor.process_env_action(action))
      dataset.append(state, action, reward, nextState=nextState)
      if done:
        dist = modelFn(np.concatenate([repmat(state, len(policyFn.actions), 1),
          np.array(policyFn.actions)], axis=1))
        lastQ += dist[policyFn.chosenAction, 0]
        lastR += reward
      state = nextState
      steps += 1
      lastSteps += 1
      if args.render and rollout % args.render_interval == 0:
        env.render()
    if stopsig:
      break

    dataset.reset() # push trajectory into the dataset buffer

    if rollout >= 1024 and rollout % 128 == 0:
      if not args.silent:
        print("Preprocessing...")
      D = dataset.sample(1024, gamma=args.gamma)
      QS0 = np.concatenate([D["states"], D["actions"]], axis=1)
      #nextActions = D["actions"]  # choosing the max action is a pain, leave that
      #                            # up to the actor in the actor-critic models
      #                            # just use the continuing action
      #Q1 = modelFn(np.concatenate([D["nextStates"], nextActions], axis=1))
      Q1 = []
      for i in range(D["nextStates"].shape[0]):
        print(i, "of", D["nextStates"].shape[0])
        if stopsig:
          break
        dist = modelFn(np.concatenate([repmat(D["nextStates"][i, :],
          allActions.shape[0], 1), allActions], axis=1))
        Q1.append(np.max(dist)) # max[a' in A](Q(s', a'))
      if stopsig:
        break
      Q1 = np.array([Q1]).T
      R = np.where(np.array([D["terminal"]]).T,
          np.array([D["rewards"]]).T,
          np.array([D["rewards"]]).T + args.gamma * Q1)

      if not args.silent:
        print("Training...")
      modelFn.fit({ "data": QS0, "label": R }, num_epochs=500)

      print("Reward:", reward if (reward >= 0.00001) else 0, "with Error:",
          modelFn.score(), "with steps:", steps, "Q,R:",
          float(lastQ) / lastSteps, float(lastR) / lastSteps)
      if args.logfile:
        log.write("[" + str(rollout) + ", " + str(steps) + ", " + str(reward) +
            ", " + str(modelFn.score()) + ", " + str(float(lastQ) / lastSteps) +
            ", " + str(float(lastR) / lastSteps) + "]\n")
      lastQ = 0
      lastR = 0
      lastSteps = 0

    rollout += 1
    if args.eps_decay and rollout % 100 == 0:
      policyFn.epsilon *= 0.95
      if policyFn.epsilon < min(0.1, args.epsilon):
        policyFn.epsilon = min(0.1, args.epsilon)
      print("Epsilon is now:", policyFn.epsilon)

  if args.logfile:
    log.close()
  if args.save_params:
    print("Saving params...")
    modelFn.save_params(args.save_params)

if __name__ == "__main__":
  main()
