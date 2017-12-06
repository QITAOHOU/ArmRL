#!/usr/bin/env python3
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("datafile",
    help="File where the data logs are stored")
args = parser.parse_args()

with open(args.datafile, "r") as fp:
  data = np.array([eval(line.strip()) for line in fp])

plt.plot(data[:, 1])
plt.show()
