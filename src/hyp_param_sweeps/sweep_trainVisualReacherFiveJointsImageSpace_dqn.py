import sys
import os
sys.path.append('../')
sys.path.append('.')

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from src.bendRL_env.VisualReacherFiveJointsImageSpace import ReacherFiveJointsImageSpace
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack
from typing import Callable


sweep_dqn_configuration = {
    "project": "ur10e_rl_project",
    "entity": "rl-team",
    "method": "grid",
    "parameters": {
        "env_name": {
            "value":  "ReacherFiveJointsImageSpace",
        },
        "env_type": {
            "values": ["static", "reaching", "tracking"],
        },
        "target_position": {
            "values": ["950,550"], #["950,550", "1700,900", "200,200", "200,900", "1700,200"],
        },
        "max_ep_len": {
            "value": 150,
        },
        "circle_rad_importance": {
           "values": [0.95, 0.5],
        },
        "shape_reward": {
            "values": [1],
        },
        "random_start": {
            "values": [0],
        },
        "record_ep_state_freq": {
            "value": 0,
        },
        "render_mode": {
            "value": "None", #"red_channel" #"human" #"None" "rgb_array"
        },
        "run": {
            "values": [1,2,3],
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
        "normalize_state": {
            "values": [0],
        },
        "total_timesteps": {
            "value": 80000,
        },
        "linear_scheduler": {
            "values": [0],
        },
        "learning_rate": {
            "values": [0.0001],
        },
        "init_learning_rate": {
            "values": [0.003],
        },
        "learning_starts": {
            "value": 1000,
        },
        "buffer_size": {
            "values": [20000],
        },
        "tau": {
            "value": 1,
        },
        "gamma": {
            "values": [0.99],
        },
        "target_update_interval": {
            "values": [300],
            },
        "train_freq": {
            "values": [1],
        },
        "gradient_steps": {
            "values": [3],
        },
        "exploration_fraction": {
            "values": [0.8],
            },
        "exploration_initial_eps": {
            "value":  1,
        },
        "exploration_final_eps": {
            "values": [0.05],
        },
        "batch_size": {
            "values": [32],
        },
        "n_stack": {
            "values": [1],
        },
        "save_details": {
            "value": 0,
        }
    }
}

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func

def train():
    start_pos_string = ["fixed_start", "random_start"]
    run = wandb.init(sync_tensorboard=True, monitor_gym=True, save_code=False)
    print(run)
    # Create log dir
    log_dir = "./" + wandb.config.env_type + "/" + wandb.config.rl_name + "/" + \
              start_pos_string[wandb.config.random_start] + "/" + "run" + \
              str(wandb.config.run) + "_" + run.id
    os.makedirs(log_dir, exist_ok=True)

    def make_env():
        pos = list(map(int, wandb.config.target_position.split(',')))
        env = DummyVecEnv([lambda: Monitor(ReacherFiveJointsImageSpace(random_start=wandb.config.random_start,
                                                            target_position=pos,
                                                            STEPS_IN_EPISODE=wandb.config.max_ep_len,
                                                            log_state_actions=False,
                                                            shape_reward=wandb.config.shape_reward,
                                                            file_name_prefix=wandb.config.rl_name + "_" + run.id,
                                                            env_type=wandb.config.env_type,
                                                            seed=(wandb.config.seed+wandb.config.run),
                                                            render_mode=wandb.config.render_mode,
                                                            rad_imp=wandb.config.circle_rad_importance,
                                                            save_state_freq=wandb.config.record_ep_state_freq),
                                           log_dir)])
        if wandb.config.normalize_state == 1:
            if wandb.config.rl_name == "DQN":
                env = VecTransposeImage(env)
            env = VecFrameStack(env, n_stack=wandb.config.n_stack)
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
        else:
            if wandb.config.rl_name == "DQN":
                env = VecTransposeImage(env)
            env = VecFrameStack(env, n_stack=wandb.config.n_stack)
            env = VecNormalize(env, norm_obs=False, norm_reward=True)
        
        if wandb.config.save_details == 1:
            stats_path = os.path.join(log_dir,
                                    "run" + str(wandb.config.run) + "_vec_normalize_" + run.id + ".pkl")
            env.save(stats_path)
        return env

    env = make_env()

    if wandb.config.save_details == 1:
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir,
                                             name_prefix= "checkpoint_run" +
                                                          str(wandb.config.run) + '_' + run.id)

    '''policy, env, learning_rate = 0.0003, n_steps = 2048, batch_size = 64, n_epochs = 10, gamma = 0.99, 
    gae_lambda = 0.95, clip_range = 0.2, clip_range_vf = None, normalize_advantage = True, ent_coef = 0.0, 
    vf_coef = 0.5, max_grad_norm = 0.5, use_sde = False, sde_sample_freq = -1, target_kl = None, 
    tensorboard_log = None, create_eval_env = False, policy_kwargs = None, verbose = 0, seed = None, 
    device = 'auto', _init_setup_model = True)'''
    #learning_rate=linear_schedule(wandb.config.init_learning_rate)
    # learning_rate = wandb.config.learning_rate

    ''' learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, batch_size=32, tau=1.0, gamma=0.99, 
    train_freq=4, gradient_steps=1, replay_buffer_class=None, replay_buffer_kwargs=None, optimize_memory_usage=False, 
    target_update_interval=10000, exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.05, '''
    # Create RL model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = None
    if wandb.config.rl_name == "DQN":
        if wandb.config.linear_scheduler == 1:
            model = DQN(wandb.config.policy_type, env, learning_rate=linear_schedule(wandb.config.init_learning_rate), verbose=1,
                        tensorboard_log="./tensorboard" + log_dir + "/", batch_size=wandb.config.batch_size,
                        buffer_size=wandb.config.buffer_size, tau=wandb.config.tau, gamma=wandb.config.gamma,
                        target_update_interval=wandb.config.target_update_interval, gradient_steps=wandb.config.gradient_steps,
                        exploration_fraction=wandb.config.exploration_fraction, learning_starts=wandb.config.learning_starts,
                        exploration_initial_eps=wandb.config.exploration_initial_eps,
                        exploration_final_eps=wandb.config.exploration_final_eps, train_freq=wandb.config.train_freq, device=device)
        else:
            model = DQN(wandb.config.policy_type, env, learning_rate=wandb.config.learning_rate, verbose=1,
                    tensorboard_log="./tensorboard" + log_dir + "/", batch_size=wandb.config.batch_size,
                    buffer_size=wandb.config.buffer_size, tau=wandb.config.tau, gamma=wandb.config.gamma,
                    target_update_interval=wandb.config.target_update_interval, gradient_steps=wandb.config.gradient_steps,
                    exploration_fraction=wandb.config.exploration_fraction, learning_starts=wandb.config.learning_starts,
                    exploration_initial_eps=wandb.config.exploration_initial_eps,
                    exploration_final_eps=wandb.config.exploration_final_eps, train_freq=wandb.config.train_freq, device=device)
            
    if wandb.config.save_details == 1:
        model.learn(
            total_timesteps=wandb.config.total_timesteps,
            callback=[WandbCallback(
                gradient_save_freq=400,
                model_save_path="models/" +  log_dir + "/",
                verbose=2,
            ), checkpoint_callback],
        )
        model.save("final_model" +  log_dir + "/")
    else:
        model.learn(
            total_timesteps=wandb.config.total_timesteps,
            callback=[WandbCallback(
                verbose=2,
            )],
        )
    env.close()


def main():
    sweep_id = wandb.sweep(sweep_dqn_configuration)
    print(sweep_id)
    # run the sweep
    wandb.agent(sweep_id, function=train)


if __name__ == "__main__":
    main()
