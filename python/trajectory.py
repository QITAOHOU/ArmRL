#!/usr/bin/env python3
import numpy as np
import physx
import simulation
import argparse
import time

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("trajectory_csv", type=str,
      help="Trajectory file as a csv of 7 angles per line")
  args = parser.parse_args()

  # grab the trajectory
  with open(args.trajectory_csv, "r") as fp:
    trajectory = [np.array([eval(n) for n in line.strip().split(",")]) \
        for line in fp]

  # connect to the visualizer (remove if unnecessary)
  arm = simulation.Arm()

  # continuously visualize the trajectory
  while True:
    for q in trajectory:
      positions = physx.forwardKinematics(arm.default_length, q)
      arm.setPositions(positions)
      time.sleep(0.02)

if __name__ == "__main__":
  main()
