# Pepper_DRL_MAExploration
Multi Agent Searching using Deep Reinforcement Learning with A3C Algorithm

This repository uses Deep Reinforcement Learning (DRL) with Imitation Learning (IL) using A3C (Asynchronous Advantage Actor Critic) Algorithm to train multiple-agents to perform Exploration of an uncharted area in a coordinated and decentralised fashion. The simulation environment is custom built entirely on ROS (Robot Operating System) and the environment is capable of:
1) Generating random maps and spawn multiple agents
2) Obtain Laser Scan around the agents and perform SLAM (Simultaneous Localization And Mapping) with gmapping
3) Communicate the built maps with the nearby agents within communication range
4) Perform point to point navigation using Global Planner
5) Perform autonmous exploration using Frontier, RRT (Rapidly exploring Random Tree) and Hector Exploraion ROS Packages, which are used as experts that generate paths to train the DRL network using Imitation Learning.

## Getting Started ##
### System Requirements: ###
1) Ubuntu 16.04 (Xenial Xerus) OS
2) ROS Kinetic Kame Framework
3) Python 2.7
4) Tensorflow 1.11

### Setting up the Environment: ###
1) Create a catkin workspace and clone all the contents in `/catkin_ws/src` directory of this repository into its source folder. And then run the `catkin_make` command.
2) Install OpenAI Gym: `pip instal gym`, and run the following commands to install the custom environment:
   ```
   cd custom_gym
   pip install -e .
   ```
   If you wish to uninstall the environment later for reasons like changing the directory of the environment, you can run the following:
   ```
   pip uninstall custom_env
   ```
   More details about installing and uninstalling OpenAI gym custom envs can be found [here](https://medium.com/analytics-vidhya/building-custom-gym-environments-for-reinforcement-learning-24fa7530cbb5).

## Environment ##
### Key Files: ###
- `/custom_gym/envs/custom_dir/Custom_Env.py` - Containing the Environment, Agents, SLAM and Navigation classes.
- `Env_sim.py` - Demonstrates the procedure to initialize the Environment and run the Key Functions in it.
- `Endurance_Test.py` - Debugging file to test if the Environment is set up correctly by initializing and resetting it several times.
- `/catkin_ws/src/pepper/launch/env_map_merge_global.launch` - ROS launch file that launches the global ROS nodes of an environment (such as global map merger, central RRT exploration planner, etc.), which will be used/shared by all the spawned agents. This launch file is initialised only once in each environment instance.
- `/catkin_ws/src/pepper/launch/gmapping_simple_sim.launch` - ROS launch file that launches all the ROS nodes specific to each agents (such as gmapping, global planner, exploration packages, etc.). This launch file is initialised everytime an agent is spawned in the environment.
- `/catkin_ws/src/pepper/src/global_map_merger.py` - Combines the maps explored by all the agents to be fed into the expert exploration packages for Imitation Learning. This ROS node is called by `env_map_merge_global.launch` and has one instance running per environment.
- `/catkin_ws/src/pepper/src/local_map_receiver.py` - This ROS node is initialized by `gmapping_simple_sim.launch` and has one instance running per agent. This node combines the maps buily by the nearby agents to an agents which are present within the defined communication range.
- `/catkin_ws/src/pepper/src/occupancy_counter.py` - This ROS node calculates the exploration progress at each step, by counting the no. of explored and unexplored pixels in a map, thereby allowing the environment to decide when exactly to terminate an episode. This node is initialized by both `env_map_merge_global.launch` and `gmapping_simple_sim.launch` to determine the exploration progress of both local and global maps.
- `/catkin_ws/src/pepper/src/frontier_detector.py` - This ROS node detects and outputs the frontier points in the gampping generated maps, which would be sent as observation into the DRL network to assist the Agents to make more informed choices on where to head to. This node is initialized by both launch files as well, to detect the frontier points in both maps.

