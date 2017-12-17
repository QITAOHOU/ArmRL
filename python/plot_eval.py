#!/usr/bin/env python3
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("datafile",
    help="File where the data logs are stored (eg. data.log)")
parser.add_argument("--descriptor",
    help="The description of these plots")
args = parser.parse_args()

with open(args.datafile, "r") as fp:
  data = np.array([eval(line.strip()) for line in fp])

descriptor = " (" + args.descriptor + ")" if args.descriptor else ""

plt.figure()
plt.plot(data[:, 1])
plt.title("Error" + descriptor)
plt.xlabel("Episode")
plt.ylabel("Error")

plt.figure()
plt.plot(data[:, 2])
plt.title("Average Q-Value" + descriptor)
plt.xlabel("Episode")
plt.ylabel("Q-Value")

plt.figure()
plt.plot(data[:, 3])
plt.title("Average Reward" + descriptor)
plt.xlabel("Epsiode")
plt.ylabel("Reward")

plt.show()
