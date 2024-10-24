import os
import sys
sys.path.append("src/") 
sys.path.append('../')
sys.path.append('.')
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from src.bendRL_env.VisualCartesianReacherFiveJointsGoal import VisualReacherFiveJoints
from src.SaveOnBestTrainingRewardCallback import SaveOnBestTrainingRewardCallback
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback


config = {
    "policy_type": "CnnPolicy",
    "render_mode": "human",
    "total_timesteps": 20000,
    "learning_rate": 0.0001,
    "learning_starts": 200,
    "buffer_size": 1000,
    "target_update_interval": 200,
    "train_freq": 4,
    "exploration_fraction": 0.9,
    "env_name": "VisualCartesianReacherFiveJoints",
    "rl_name": "PPO",
    "goal_threshold": 0.2,
    "env_type": "static",
    "file_name_prefix": "realsense",
    "random_start": 0,
    "n_steps": 256,  # this is used with PPO and A2C,
    "batch_size": 32,  # this is used with PPO
    "shape_reward": 0
}

log_dir = "VisualCartesianReacherFiveJoints_realsense/"
os.makedirs(log_dir, exist_ok=True)


env = VisualReacherFiveJoints(random_start=config["random_start"],
                                                        log_state_actions=True,
                                                        goal_threshold=config["goal_threshold"],
                                                        file_name_prefix=config["file_name_prefix"],
                                                        render_mode=config["render_mode"])


model = PPO(config["policy_type"],env, verbose=1,
                tensorboard_log="./tensorboard/"+config["file_name_prefix"]+"_run/",
                n_steps=config["n_steps"], batch_size=config["batch_size"])


checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                            name_prefix=config["file_name_prefix"]+"_run")



# Train the agent
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=checkpoint_callback
)
# model.save("ppo2_bender")
# model.save("a2c_bender")
model.save(config["file_name_prefix"]+"_run_visCartesian_bender")
env.close()




# if __name__ == "__main__":
#     main()


