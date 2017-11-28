# Homework 2

Basketball Reinforcement Learning, Timothy Yong.

## Installation Requirements

```
sudo apt-get install cmake
sudo apt-get install gazebo9
sudo apt-get install libsdformat6-dev #(if there is no such package, install from source)
sudo apt-get install libboost-all-dev
sudo apt-get install libarmadillo-dev
sudo apt-get install libprotobuf-dev
sudo apt-get install python3-pip
sudo pip3 install numpy pygame
```

Note that installing gazebo/sdf have seen issues in the past. While this
particular project uses gazebo9, it should work fine with gazebo8. To use with
other versions of sdf, change the sdf format from (1.6) to (1.x), the version of
sdf you wish to use, in several of the files (eg. `basketball_world.world` and
`arm_gz.cc`).

## Running this project

In order to run this project, first compile the necessary libraries:

```
mkdir build && cd build
cmake ..
make
```

Now you can run the project. First start by running the gazebo simulator used
for visualization:

```
./scripts/start_viz.sh
```

Open a new terminal and navigate to the project directory. Then type:

```
cd python
./twojoint.py --render
```

If you want to stop visualization, run the following:
```
./killserver.sh
```
