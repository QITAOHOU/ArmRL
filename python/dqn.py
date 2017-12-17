#!/usr/bin/env python3
import numpy as np
from numpy.matlib import repmat
import argparse
import random
import time
import os
import signal
from console_widgets import ProgressBar
from environment import BasketballVelocityEnv
from core import ContinuousSpace, DiscreteSpace, DQNProcessor
from models import DQNNetwork
from policy import EpsilonGreedyPolicy
from memory import RingBuffer

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
  parser.add_argument("--eps_anneal", type=int, default=0,
      help="The amount of episodes to anneal epsilon by")
  parser.add_argument("--sample_size", type=int, default=256,
      help="Number of samples from the dataset per episode")
  parser.add_argument("--num_epochs", type=int, default=50,
      help="Number of epochs to run per episode")
  parser.add_argument("--episode_length", type=int, default=128,
      help="Number of rollouts per episode")
  parser.add_argument("--noise", type=float,
      help="Amount of noise to add to the actions")
  parser.add_argument("--test", action="store_true",
      help="Test the params")
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
  modelFn = DQNNetwork(
      sizes=[stateSpace.n + actionSpace.n, 128, 256, 256, 128, 1],
      alpha=0.001, use_gpu=True, momentum=0.9)
  if args.load_params:
    print("Loading params...")
    modelFn.load_params(args.load_params)

  allActions = actionSpace.sampleAll()
  policyFn = EpsilonGreedyPolicy(epsilon=args.epsilon if not args.test else 0,
      getActionsFn=lambda state: allActions,
      distributionFn=lambda qstate: modelFn(qstate),
      processor=processor)
  replayBuffer = RingBuffer(max_limit=2048)
  if args.logfile:
    log = open(args.logfile, "a")

  if not args.silent:
    print("Env space range:", env.state_range())
    print("Env action range:", env.action_range())
    print("State space:", stateSpace.n)
    print("Action space:", actionSpace.n)
    print("Action space bins:", actionSpace.bins)
    print("Epsilon:", args.epsilon)
    print("Epsilon anneal episodes:", args.eps_anneal)
    print("Gamma:", args.gamma)
    __actionShape = policyFn.getActions(None).shape
    totalActions = np.prod(actionSpace.bins)
    print("Actions are sampled:", __actionShape[0] != totalActions)
    print("Number of actions:", totalActions)

  rollout = 0
  if not args.silent and not args.test:
    iterationBar = ProgressBar(maxval=args.episode_length)
  while args.num_rollouts == -1 or rollout < args.num_rollouts:
    if stopsig: break
    if not args.silent and not args.test:
      iterationBar.printProgress(rollout % args.episode_length,
          prefix="Query(s,a,s',r)", suffix="epsilon: " + str(policyFn.epsilon))
    state = env.reset()
    reward = 0
    done = False
    steps = 0
    while not done and steps < 5: # 5 step max
      action = policyFn(state)
      if steps == 4: # throw immediately
        action[-2] = 0
        action[-1] = 1
      envAction = processor.process_env_action(action)
      if args.noise:
        envAction[:7] += np.random.normal(scale=np.ones([7]) * args.noise)
      nextState, reward, done, info = env.step(envAction)
      replayBuffer.append([state, action, nextState, reward, done])
      if args.test and done: print("Reward:", reward)
      state = nextState
      steps += 1
      if args.render and (rollout + 1) % args.render_interval == 0:
        env.render()

    rollout += 1
    if args.eps_anneal > 0: # linear anneal
      epsilon_diff = args.epsilon - min(0.1, args.epsilon)
      policyFn.epsilon = args.epsilon - min(rollout, args.eps_anneal) / \
          float(args.eps_anneal) * epsilon_diff

    if rollout % args.episode_length == 0 and not args.test:
      dataset = replayBuffer.sample(args.sample_size)
      states = np.array([d[0] for d in dataset])
      actions = np.array([d[1] for d in dataset])
      nextStates = [d[2] for d in dataset]
      rewards = np.array([[d[3]] for d in dataset]) # rewards require extra []
      terminal = [d[4] for d in dataset]

      QS0 = processor.process_Qstate(states, actions)
      Q1 = np.zeros(rewards.shape, dtype=np.float32)
      progressBar = ProgressBar(maxval=len(nextStates))
      for i, nextState in enumerate(nextStates):
        if stopsig: break
        if not args.silent:
          progressBar.printProgress(i, prefix="Creating Q(s,a)",
              suffix="%s / %s" % (i + 1, len(nextStates)))
        if terminal[i]: continue # 0
        dist = modelFn(processor.process_Qstate(
          repmat(nextState, allActions.shape[0], 1), allActions))
        Q1[i, 0] = np.max(dist) # max[a' in A]Q(s', a')
      if stopsig: break
      Q0_ = rewards + args.gamma * Q1
      modelFn.fit({ "qstates": QS0, "qvalues": Q0_ },
          num_epochs=args.num_epochs)

      avgQ = np.sum(Q0_) / Q0_.shape[0]
      avgR = np.sum(rewards) / rewards.shape[0]
      print("Rollouts:", rollout,
          "Error:", modelFn.score(),
          "Average Q:", avgQ,
          "Average R:", avgR,
          "\n")
      if args.logfile:
        log.write("[" + str(rollout) + ", " +
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
