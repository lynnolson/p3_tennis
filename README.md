# Collaboration and Competition 

### Introduction

This project uses Unity’s Tennis environment which simulates two agents playing tennis.  Each agent has a "racket" that it uses to hit a "ball" over a net.

Each agent gets its own copy of the environment. The environment is defined by a continuous state space with dimension size of 24 corresponding to the position and velocity of the ball and racket.  The Introduction to the Environment says there are 8 variables, though apparently three sets of these get stacked to create the final 24 for each agent playing tennis.

The action space is also continuous with a dimension of 2 corresponding an agent's movement toward or away from the net along with jumping.

The goal of each agent is to keep the ball in play.  A reward of 0.1 is given if the agent hits the ball over the net.  In contrast, a reward of -0.01 is given if the ball hits the ground or goes out of bounds.

The environment is considered solved when the maximum score between the two agents is at least 0.5 averaged over a series of 100 episodes.

The code is a modified form of the ddpg-pendulum code provided by Udacity in their Deep Reinforcement Learning Nanodegree Program and more directly from the same code I used in Project 2, Continuous Control (https://github.com/lynnolson/p2_continuous-control)

### Getting Started
1. [Download](https://www.python.org/downloads/) and install Python 3.6 or higher if not already installed.
2. Install conda if not already installed.  To install conda, follow these [instructions](https://conda.io/docs/user-guide/install/index.html)
3. Create and activate a new conda environment
```
conda create -n p3_tennis python=3.6
conda activate p3_tennis
```
3. Clone this GitHub repository
```
git clone https://github.com/lynnolson/p3_tennis.git
```
4. Change to the p3_tennis directory and install python dependencies by running setup.py
```
cd p3_tennis
python setup.py install
```
5. Download the Unity Tennis environment from the link below.  Note: The training procedure has been tested on Linux with a headless environment (the first option below) and on MacOSX (the second option.)
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

6. Place the environment zip in the `p3_tennis/` folder, and unzip (or decompress) the file.

### Training Instructions

To train the agent, run train.py

```
python train.py -tennis_env_path Tennis.app -ckpt_path_prefix checkpoint
```

To train on Linux, use

```
python train.py -tennis_env_path Tennis_Linux_NoVis/Tennis.x86_64 -ckpt_path_prefix checkpoint
```

To save a plot of the scores over time (successive episodes), set the argument plot_path to a specific file

```
python train.py -tennis_env_path Tennis.app -ckpt_path_prefix checkpoint -plot_path score_per_episode.png
```
The model weights are saved in two files prefixed by ckpt_path_prefix - one corresponds to actor's network weights and the other to the critic's.  Currently there is no mechanism to recreate the model from these parameters.
When you are done, deactivate the conda environment:
```
conda deactivate
```
### Note
The whole procedure above has been tested on Ubuntu 16.04 and OS X El Capitan.
