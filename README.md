# bendRL: Visual Reaching and tracking

In this work, we evaluate the ability to learn image-based reaching and tracking asks via reinforcement learning on the UR10e robotic arm by Universal Robots

![plot](figures/interactive_env.png)



## Objective

This work explores the potential of on- and off-policy reinforcement learning algorithms to perform image-based reaching and tracking tasks. In order to guage performance in a realistic manner, the physical training environment contains variable light conditions and numerous objects and colours that may distract the agent. All training is performed end-to-end and online. 

## Installation


## Reproducibility 


## Results

The plots below show the learning curves for the DQN and PPO agents. PPO learns to achieve a much higher mean reward than DQN in the first 40k time steps. After an additional 20k of training steps DQN approachs PPO on reacher and tracker but remains significantly worse on the static reacher task.

![plot](figures/paper_learning_curves.png)
