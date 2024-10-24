import gym
import numpy as np
import pandas as pd
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
from VisualReacherFiveJointsImageSpace import ReacherFiveJointsImageSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage, VecFrameStack
import yaml

sweep_dqn_configuration = {
    "project": "ur10e_rl_project",
    "entity": "rl-team",
    "method": "grid",
    "parameters": {
        "env_name": {
            "value":  "ReacherFiveJointsImageSpace",
        },
        "env_type": {
            "value": "static",
        },
        "target_position": {
            "value": "950,550", #["950,550", "1700,900", "200,200", "200,900", "1700,200"],
        },
        "max_ep_len": {
            "value": 150,
        },
        "normalize_state": {
            "value": 0,
        },
        "circle_rad_importance": {
            "value": 0.5,
        },
        "shape_reward": {
            "value": 1,
        },
        "random_start": {
            "value": 0,
        },
        "record_ep_state_freq": {
            "value": 0,
        },
        "render_mode": {
            "value": "human_reward", #"red_channel" #"human" #"None" "rgb_array"
        },
        "run": {
            "value": 1,
        },
        "seed": {
            "value": 1234,
        },
        "rl_name": {
            "value": "DQN",
        },
        "policy_type": {
            "value": "CnnPolicy",
        },
        "total_timesteps": {
            "value": 15000,
        },
        "learning_rate": {
            "value": 0.005,
        },
        "init_learning_rate": {
            "value": 0.001,
        },
        "learning_starts": {
            "value": 1000,
        },
        "buffer_size": {
            "value": 2500, #[1000,2500],
        },
        "tau": {
            "value": 1,
        },
        "gamma": {
            "value": 0.98, #[0.98, 0.99],
        },
        "target_update_interval": {
            "value": 50, #[50, 100],
            },
        "train_freq": {
            "value": 5, #[5,10],
        },
        "gradient_steps": {
            "value": 8, #[1,8],
        },
        "exploration_fraction": {
            "value": 0.1, # [0.1, 0.3, 0.6],
            },
        "exploration_initial_eps": {
            "value":  1,
        },
        "exploration_final_eps": {
            "value": 0.01, #[0.01, 0.05],
        },
        "batch_size": {
            "value": 32, #[32,64],
        },
        "n_stack": {
            "value": 4,
        },
        "save_details": {
            "value": 0,
        }
    }
}

pos = list(map(int, sweep_dqn_configuration['parameters']['target_position']['value'].split(',')))
env = DummyVecEnv([lambda: Monitor(ReacherFiveJointsImageSpace(random_start=sweep_dqn_configuration['parameters']['random_start']['value'],
                                                            target_position=pos,
                                                            STEPS_IN_EPISODE=sweep_dqn_configuration['parameters']['max_ep_len']['value'],
                                                            log_state_actions=False,
                                                            shape_reward=sweep_dqn_configuration['parameters']['shape_reward']['value'],
                                                            file_name_prefix=sweep_dqn_configuration['parameters']['rl_name']['value'] + "_tst",
                                                            env_type=sweep_dqn_configuration['parameters']['env_type']['value'],
                                                            seed=123,
                                                            render_mode=sweep_dqn_configuration['parameters']['render_mode']['value'],
                                                            rad_imp=sweep_dqn_configuration['parameters']['circle_rad_importance']['value'],
                                                            save_state_freq=sweep_dqn_configuration['parameters']['record_ep_state_freq']['value']),
                                           )])

if sweep_dqn_configuration['parameters']['normalize_state']['value'] == 1:
    if sweep_dqn_configuration['parameters']['rl_name']['value'] == "DQN":
        env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=sweep_dqn_configuration['parameters']['n_stack']['value'])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
else:
    env = VecFrameStack(env, n_stack=sweep_dqn_configuration['parameters']['n_stack']['value'])

model = DQN("CnnPolicy", env, verbose=1, buffer_size= sweep_dqn_configuration['parameters']['buffer_size']['value'])
model.learn(total_timesteps=100, log_interval=4)