#!/usr/bin/env python3
import numpy as np
from environment import BasketballVelocityEnv
import argparse
import time

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("trajectory_csv", type=str,
      help="Trajectory file as a csv of 7 angle vel per line")
  args = parser.parse_args()

  # grab the trajectory
  with open(args.trajectory_csv, "r") as fp:
    trajectory = [np.array([eval(n) for n in line.strip().split(",")]) \
        for line in fp]

  # connect to the visualizer (remove if unnecessary)
  env = BasketballVelocityEnv(initialLengths=[1 for i in range(7)],
      fps=60.0, timeInterval=1.0)

  # continuously visualize the trajectory
  while True:
    state = env.reset()
    reward = 0
    done = False
    for q in trajectory:
      action = list(q) + [0]
      nextState, reward, done, _ = env.step(action)
      env.render()

if __name__ == "__main__":
  main()
