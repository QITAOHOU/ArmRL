#!/usr/bin/env python3
import numpy as np
import argparse
import signal
from functools import reduce
from environment import BasketballVelocityEnv
from core import ContinuousSpace
from models import PoWERDistribution
import memory
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
  parser.add_argument("--num_rollouts", type=int, default=1000,
      help="Number of max rollouts")
  parser.add_argument("--logfile", type=str,
      help="Indicate where to save rollout data")
  parser.add_argument("--load_params", type=str,
      help="Load previously learned parameters from [LOAD_PARAMS]")
  parser.add_argument("--save_params", type=str,
      help="Save learned parameters to [SAVE_PARAMS]")
  parser.add_argument("--gamma", type=float, default=0.99,
      help="Discount factor")
  parser.add_argument("--test", action="store_true",
      help="Test the params")
  args = parser.parse_args()

  signal.signal(signal.SIGINT, stopsigCallback)
  global stopsig

  # create the basketball environment
  env = BasketballVelocityEnv(fps=60.0, timeInterval=0.1,
      goal=[0, 5, 0],
      initialLengths=np.array([0, 0, 1, 1, 1, 0, 1]),
      initialAngles=np.array([0, 45, -20, -20, 0, -20, 0]))

  # create space
  stateSpace = ContinuousSpace(ranges=env.state_range())
  actionSpace = ContinuousSpace(ranges=env.action_range())

  # create the model and policy functions
  modelFn = PoWERDistribution(stateSpace.n, actionSpace.n,
      sigma=5.0 if not args.test else 0)
  if args.load_params:
    print("Loading params...")
    modelFn.load_params(args.load_params)

  replayBuffer = ReplayBuffer(1024)
  if args.logfile:
    log = open(args.logfile, "a")
  
  rollout = 0
  while args.num_rollouts == -1 or rollout < args.num_rollouts:
    print("Iteration:", rollout)
    state = env.reset()
    reward = 0
    done = False
    steps = 0
    while not done and steps < 5:
      if stopsig:
        break
      action, eps = modelFn.predict(state,
          replayBuffer.sample(gamma=args.gamma))
      if steps == 4:
        action[-1] = 1.0
      nextState, reward, done, info = env.step(action)
      replayBuffer.append(state, action, reward, nextState=nextState,
          info={"eps": eps})
      state = nextState
      steps += 1
      if args.render and rollout % args.render_interval == 0:
        env.render()
    if stopsig:
      break

    # no importance sampling, implement it when we have small datasets
    replayBuffer.reset()
    dataset = replayBuffer.sample(gamma=args.gamma)
    modelFn.fit(dataset)

    avgR = np.sum(dataset["rewards"]) / float(len(dataset["rewards"]))
    avgQ = np.sum(dataset["values"]) / float(len(dataset["values"]))
    print("Rollouts:", rollout,
        "Error:", modelFn.score(),
        "Average Q", avgQ,
        "Average R", avgR)
    if args.logfile:
      log.write("[" + str(rollout) + ", " +
          str(modelFn.score()) + ", " +
          str(avgQ) + ", " +
          str(avgR) + "]\n")
    rollout += 1

  if args.logfile:
    log.close()
  if args.save_params:
    print("Saving params...")
    modelFn.save_params(args.save_params)

if __name__ == "__main__":
  main()
