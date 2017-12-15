#!/usr/bin/env python3
import numpy as np
from numpy.matlib import repmat
import argparse
import random
import time
import os
import signal
from console_widgets import Progbar
from environment import BasketballAccelerationEnv, BasketballVelocityEnv
from core import ContinuousSpace, \
                 DiscreteSpace, \
                 DQNProcessor
from models import DQNNetwork
from policy import EpsilonGreedyPolicy
from memory import ReplayBuffer, RingBuffer

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
  parser.add_argument("--gamma", type=float, default=0.99,
      help="Discount factor")
  parser.add_argument("--epsilon", type=float, default=0.1,
      help="Random factor (for Epsilon-greedy)")
  parser.add_argument("--eps_decay", action="store_true",
      help="Let epsilon become linearly annealed over time")
  parser.add_argument("--sample_size", type=int, default=256,
      help="Number of samples from the dataset per episode")
  parser.add_argument("--num_epochs", type=int, default=50,
      help="Number of epochs to run per episode")
  args = parser.parse_args()

  signal.signal(signal.SIGINT, stopsigCallback)
  global stopsig

  # create the basketball environment
  env = BasketballVelocityEnv(fps=60.0, timeInterval=1.0,
      goal=[0, 5, 0],
      initialLengths=np.array([0, 0, 1, 1, 1, 0, 1]),
      initialAngles=np.array([0, 45, -20, -20, 0, -20, 0]))

  # create space
  stateSpace = ContinuousSpace(ranges=env.state_range())
  actionSpace = DiscreteSpace(intervals=[25 for i in range(7)] + [1],
      ranges=env.action_range())
  processor = DQNProcessor(actionSpace)

  # create the model and policy functions
  #modelFn = DQNNetwork(sizes=[1, 128, 128, 64, 512, 256, 1],
  modelFn = DQNNetwork(
      sizes=[stateSpace.n + actionSpace.n, 128, 128, 64, 512, 256, 1],
      alpha=0.001, use_gpu=True, momentum=0.9)
  if args.load_params:
    print("Loading params...")
    modelFn.load_params(args.load_params)

  allActions = actionSpace.sampleAll()
  policyFn = EpsilonGreedyPolicy(epsilon=args.epsilon,
      getActionsFn=lambda state: allActions,
      #getActionsFn=lambda state: actionSpace.sample(2048),
      distributionFn=lambda qstate: modelFn(qstate),
      processor=processor)
  dataset = RingBuffer(max_limit=2048)
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
    __actionShape = policyFn.getActions(None).shape
    totalActions = np.prod(actionSpace.bins)
    print("Actions are sampled:", __actionShape[0] != totalActions)
    print("Number of actions:", totalActions)

  rollout = 0
  while args.num_rollouts == -1 or rollout < args.num_rollouts:
    if stopsig: break
    if not args.silent:
      print("Iteration:", rollout, "with epsilon:", policyFn.epsilon)
    state = env.reset()
    reward = 0
    done = False
    steps = 0
    while not done and steps < 5: # Raghav uses 5 step max
      action = policyFn(state)
      if steps == 4: # throw immediately
        action[-2] = 0
        action[-1] = 1
      print("A:", processor.process_env_action(action))
      nextState, reward, done, info = env.step(
          processor.process_env_action(action))
      print("R:", reward)
      print("steps:", steps)
      dataset.append([state, action, nextState, reward, done])
      state = nextState
      steps += 1
      if args.render and rollout % args.render_interval == 0:
        env.render()

    rollout += 1
    if args.eps_decay: # linear anneal
      epsilon_diff = args.epsilon - min(0.1, args.epsilon)
      policyFn.epsilon = args.epsilon - min(rollout, 1e4) / 1e4 * epsilon_diff

    if rollout % 128 == 0:
      D = dataset.sample(args.sample_size)
      states = np.array([d[0] for d in D])
      actions = np.array([d[1] for d in D])
      nextStates = [d[2] for d in D]
      rewards = np.array([[d[3]] for d in D]) # rewards require an extra []
      terminal = [d[4] for d in D]

      QS0 = processor.process_Qstate(states, actions)
      Q1 = np.zeros(rewards.shape, dtype=np.float32)
      progressBar = Progbar(maxval=len(nextStates), prefix="Generate Q(s,a)")
      for i, nextState in enumerate(nextStates):
        if stopsig: break
        if not args.silent: progressBar.printProgress(i)
        if terminal[i]: continue
        dist = modelFn(processor.process_Qstate(
          repmat(nextState, allActions.shape[0], 1), allActions))
        Q1[i, 0] = np.max(dist) # max[a' in A]Q(s', a')
      Q0_ = rewards + args.gamma * Q1
      modelFn.fit({ "qstates": QS0, "qvalues": Q0_ },
          num_epochs=args.num_epochs)

      avgQ = np.sum(Q0_) / Q0_.shape[0] # todo: change to what the nn outputs
      avgR = np.sum(rewards) / rewards.shape[0]
      print("Reward:", reward,
          "with Error:", modelFn.score(),
          "with steps:", steps, "\n",
          "Average Q:", avgQ,
          "Average R:", avgR)
      if args.logfile:
        log.write("[" + str(rollout) + ", " +
            str(steps) + ", " +
            str(modelFn.score()) + ", " +
            str(avgQ) + ", " +
            str(avgR) + "]\n")

  if args.logfile:
    log.close()
  if args.save_params:
    print("Saving params...")
    modelFn.save_params(args.save_params)

if __name__ == "__main__":
  main()
