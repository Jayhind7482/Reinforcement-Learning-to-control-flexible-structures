# Reinforcement Learning for Control of a Flexible Structure in 2D Space

This repository contains code for a project utilizing Reinforcement Learning (RL) to train an agent that controls the tip of a flexible structure in a 2D space. The primary objective of the agent is to predict the necessary movements to make the tip of the structure trace randomly generated target points within the space.

## Project Overview

In this project, a flexible structure, such as a robotic arm or similar mechanism, is modeled and controlled using RL techniques. The agent is trained to move the structure's tip to accurately follow a series of randomly generated points in a 2D plane. This involves continuous adjustments based on feedback from the environment and the predictions made by the RL model.

### Key Features:
- *Environment*: The simulation environment models a flexible structure that operates in 2D space.
- *Agent*: The RL agent is trained using reward-based learning to minimize the error between the tip's position and the target points.
- *Goal*: The trained agent is tasked with predicting actions that control the structure’s movement to trace the target points accurately.

## Getting Started

### Prerequisites:
- Python 3.8 or higher
- Reinforcement learning libraries: Stable Baselines, gymnasium , numpy , random , Scipy , math , optuna
- Simulation and visualization libraries: Matplotlib, NumPy, pygame etc.

### Installation:
1. Clone this repository:
   bash
   git clone https://github.com/yourusername/flexible-structure-rl.git
   cd flexible-structure-rl
   
2. Install the required packages:
   bash
   pip install -r requirements.txt
   

### Usage:
1. Run the following script to  run simulation environment ,train the RL agent and test the agent:
   bash
   Elastica_length6_energy_curvature_New.ipynb
   


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
