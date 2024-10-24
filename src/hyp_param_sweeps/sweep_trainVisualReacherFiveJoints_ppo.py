import sys
import os
sys.path.append('../')
sys.path.append('.')
                
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from src.bendRL_env.VisualCartesianReacherFiveJointsGoal import VisualReacherFiveJoints
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

sweep_configuration = {
    "project": "ur10e_rl_project",
    "entity": "rl-team",
    "method": "grid",
    "parameters": {
        "policy_type": {
            "value": "CnnPolicy",
            },
        "total_timesteps": {
            "value": 20000,
            },
        "learning_rate": {
            "value": 0.0001,
            },
        "learning_starts": {
            "value": 1000,
            },
        "buffer_size": {
            "value": 2000,
            },
        "target_update_interval": {
            "value": 200,
            },
        "train_freq": {
            "value": 4,
            },
        "exploration_fraction": {
            "value": 0.9,
            },
        "env_name": {
            "value":  "VisualBenderFiveJoints",
            },
        "rl_name": {
            "value": "PPO",
            # "value": DQN,
            },
        "goal_threshold": {
            "value": 0.2,
             },
        "file_name_prefix": {
            "value": "ppo_visual_threshold_point2",
            },
        "n_steps": {
            "value": 256,
            },
        "batch_size": {
            "value": 32,
            },
        "run": {
            "values": [1, 2, 3]
        },
        "n_stack": {
            "values": [1],
        },
        "random_start": {
            "value": 0,
            }
    }
}


def train():
    start_pos_string = ["fixed_start", "random_start"]
    run = wandb.init(sync_tensorboard=True, monitor_gym=True, save_code=False)
    print(run)
    run_num = wandb.config.run
    random_start = wandb.config.random_start
    # Create log dir
    log_dir = "./" + wandb.config.file_name_prefix + \
              start_pos_string[random_start] + \
              "_run" + str(run_num) + "_llc_fiveJoints_" + run.id
    os.makedirs(log_dir, exist_ok=True)

    # def make_env():
    #     env = DummyVecEnv([lambda: Monitor(VisualReacherFiveJoints(random_start=random_start,
    #                                                         log_state_actions=False,
    #                                                         goal_threshold=wandb.config.goal_threshold,
    #                                                         file_name_prefix=wandb.config.file_name_prefix),
    #                                        log_dir)])
        
    #     env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.)
    #     print("here-------------------")
    #     # env = Monitor(env, log_dir)
    #     stats_path = os.path.join(log_dir,
    #                               wandb.config.file_name_prefix + "_run" + str(run_num) + "_vec_normalize_" + run.id + ".pkl")
    #     env.save(stats_path)
    #     return env
    
    def make_env():
        env = DummyVecEnv([lambda: Monitor(VisualReacherFiveJoints(random_start=random_start,
                                                            log_state_actions=False,
                                                            goal_threshold=wandb.config.goal_threshold,
                                                            file_name_prefix=wandb.config.file_name_prefix),
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
    print(env)
    
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                             name_prefix= wandb.config.file_name_prefix + "_run" +
                                                          str(run_num) + '_model_' + run.id)

    # Create RL model
    model = None
    if wandb.config.rl_name == "PPO":
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/" + wandb.config.file_name_prefix
                                    + "_run"+str(run_num)+"_llc_fiveJoints_"+run.id+"/",
                    n_steps=wandb.config.n_steps, batch_size=wandb.config.batch_size)
    elif wandb.config.rl_name == "A2C":
        model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/" + wandb.config.file_name_prefix
                                                                 + "_run" + str(run_num) + "_llc_fiveJoints_" + run.id + "/",
                    n_steps=wandb.config.n_steps)
    elif wandb.config.rl_name == "DQN":
        model = DQN(wandb.config.policy_type, env, learning_rate=wandb.config.learning_rate, verbose=1,
                    tensorboard_log="./tensorboard/" + wandb.config.file_name_prefix
                                    + "_run"+str(run_num)+"_llc_fiveJoints_"+run.id+"/",
                    buffer_size=wandb.config.buffer_size,
                    target_update_interval=wandb.config.target_update_interval,
                    exploration_fraction=wandb.config.exploration_fraction,
                    train_freq=wandb.config.train_freq,
                    learning_starts=wandb.config.learning_starts)

    # Train the agent
    model.learn(
        total_timesteps=wandb.config.total_timesteps,
        callback=[WandbCallback(
            gradient_save_freq=100,
            model_save_path="models/" + wandb.config.file_name_prefix +"_run" +
                            str(run_num) + "_llc_fiveJoints_"+run.id+"/" + str({run.id}),
            verbose=2,
        ), checkpoint_callback],
    )

    model.save(wandb.config.file_name_prefix+"_run"+str(run_num)+"_llc_fiveJoints_bender_"+run.id)
    env.close()


def main():
    sweep_id = wandb.sweep(sweep_configuration)
    print(sweep_id)
    # run the sweep
    wandb.agent(sweep_id, function=train)


if __name__ == "__main__":
    main()