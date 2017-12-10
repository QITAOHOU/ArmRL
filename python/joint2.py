#!/usr/bin/env python3
import numpy as np
from numpy.matlib import repmat
import argparse
import random
import time
import os
import signal
from environment import BasketballAccelerationEnv
from core import ContinuousSpace, \
                 DiscreteSpace, \
                 JointProcessor
from models import MxFullyConnected
from policy import EpsilonGreedyPolicy
from memory import ReplayBuffer

def createAction(mlaction):
  joint1 = mlaction[0]
  joint2 = mlaction[1]
  release = mlaction[2]
  return np.array([0, joint1, joint2, 0, 0, 0, 0, release],
      dtype=np.float32)

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
  env = BasketballAccelerationEnv(fps=60.0, timeInterval=0.1,
      goal=[0, 5, 0],
      initialLengths=np.array([0, 0, 1, 1, 0, 0, 0]),
      initialAngles=np.array([0, 45, 0, 0, 0, 0, 0]))

  # create space
  stateSpace = ContinuousSpace(ranges=env.state_range())
  actionRange = env.action_range()
  actionSpace = DiscreteSpace(intervals=[15 for i in range(2)] + [1],
      ranges=[actionRange[1],
              actionRange[2],
              actionRange[7]])
  processor = JointProcessor(actionSpace)

  # create the model and policy functions
  modelFn = MxFullyConnected(sizes=[stateSpace.n + actionSpace.n,
    1024, 1024, 1], alpha=0.001, use_gpu=True)
  if args.load_params:
    print("loading params...")
    modelFn.load_params(args.load_params)

  softmax = lambda s: np.exp(s) / np.sum(np.exp(s))
  allActions = actionSpace.sampleAll()
  policyFn = EpsilonGreedyPolicy(epsilon=args.epsilon,
      getActionsFn=lambda state: allActions,
      distributionFn=lambda qstate: softmax(modelFn(qstate)))
  dataset = ReplayBuffer()
  if args.logfile:
    log = open(args.logfile, "a")

  rollout = 0
  while args.num_rollouts == -1 or rollout < args.num_rollouts:
    print("Iteration:", rollout)
    state = env.reset()
    reward = 0
    done = False
    steps = 0
    lastQ = 0
    while not done:
      if stopsig:
        break
      action = policyFn(state)
      nextState, reward, done, info = env.step(createAction(
          processor.process_env_action(action)))
      dataset.append(state, action, reward, nextState)
      if done:
        dist = modelFn(np.concatenate([repmat(state, len(policyFn.actions), 1),
          np.array(policyFn.actions)], axis=1))
        lastQ = dist[policyFn.chosenAction, 0]
      state = nextState
      steps += 1
      if args.render and rollout % args.render_interval == 0:
        env.render()
    if stopsig:
      break

    dataset.reset() # push trajectory into the dataset buffer

    print("Training...")
    D = dataset.sample(1024, gamma=args.gamma)
    QS0 = np.concatenate([D["states"], D["actions"]], axis=1)
    #nextActions = D["actions"]  # choosing the max action is a pain, leave that
    #                            # up to the actor in the actor-critic models
    Q1 = []
    for i in range(D["nextStates"].shape[0]):
      dist = modelFn(np.concatenate([repmat(D["nextStates"][i, :],
        allActions.shape[0], 1), allActions], axis=1))
      Q1.append(np.max(dist))
    #Q1 = modelFn(np.concatenate([D["nextStates"], nextActions], axis=1))
    R = np.where(np.array([D["terminal"]]).T,
        np.array([D["rewards"]]).T,
        np.array([D["rewards"]]).T + args.gamma * np.array([Q1]).T)
    modelFn.fit({ "data": QS0, "label": R })

    print("Reward:", reward if (reward >= 0.00001) else 0, "with Error:",
        modelFn.score(), "with steps:", steps, "with last Q:", lastQ)
    if args.logfile:
      log.write("[" + str(rollout) + ", " + str(steps) + ", " + str(reward) +
          ", " + str(modelFn.score()) + ", " + str(lastQ) + "]\n")

    rollout += 1
    if args.eps_decay and rollout % 100 == 0:
      policyFn.epsilon *= 0.95
      if policyFn.epsilon < 0.1:
        policyFn.epsilon = 0.1
      print("Epsilon is now:", policyFn.epsilon)

  if args.logfile:
    log.close()
  if args.save_params:
    print("saving params...")
    modelFn.save_params(args.save_params)

if __name__ == "__main__":
  main()
