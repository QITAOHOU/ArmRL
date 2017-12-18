# Homework 2

Basketball Reinforcement Learning, Timothy Yong.

## Installation Requirements

```
sudo apt-get install cmake
sudo apt-get install gazebo9
sudo apt-get install libsdformat6-dev
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
`ball_gz.cc`). For any of the above dependencies which do not exist, please try
to install them from source.

## Running this project

In order to run this project, first compile the necessary libraries:

```
mkdir build && cd build
cmake ..
make
```

## Running DQN

To run the standard DQN (and to save the model after training), just type the
following:

```
cd python
./dqn --save_params newmodel.param --num_rollouts 12800
```

To view all the flags which you can set, type:

```
./dqn --help
```

## Running DQN on the simulation

Now you can run the project. First start by running the gazebo simulator used
for visualization:

```
./scripts/start_armviz.sh
```

Open a new terminal and navigate to the project directory. To run the DQN model:

```
cd python
./dqn --render
```

If you want to stop visualization, run the following:
```
./killserver.sh
```

## Running PoWER

To run PoWER (and to save the model after training), just type the following:

```
cd python
./power --save_params newmodel.param --num_rollouts 1000
```
