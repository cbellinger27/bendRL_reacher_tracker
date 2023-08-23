import os
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from VisualCartesianReacherFiveJointsGoal import VisualReacherFiveJoints



config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 20000,
    "learning_rate": 0.0001,
    "learning_starts": 200,
    "buffer_size": 1000,
    "target_update_interval": 200,
    "train_freq": 4,
    "exploration_fraction": 0.9,
    "env_name": "VisualBenderFiveJoints",
    "rl_name": "PPO",
    "goal_threshold": 0.2,
    "file_name_prefix": "ppo_smaller_threshold_point2",
    "n_steps": 256, # this is used with PPO and A2C,
    "batch_size": 32 # this is used with PPO
}

run_num = 0

run = wandb.init(
    project=config["env_name"],
    entity="nrc-rl-robotics",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


# Create log dir
log_dir = "./" + config["file_name_prefix"] + "_run" + str(run_num) + "_llc_fiveJoints_" + run.id
os.makedirs(log_dir, exist_ok=True)

def make_env():
    # env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
    env = DummyVecEnv([lambda: Monitor(VisualReacherFiveJoints(random_start=0, log_state_actions=False,
                                                                goal_threshold=config["goal_threshold"],
                                                                file_name_prefix=config["file_name_prefix"],
                                                                save_images=False,
                                                              ),
                                       log_dir)])
    # env = BenderFourJoints(random_start=0, log_state_actions=True)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.)
    # env = Monitor(env, log_dir)
    stats_path = os.path.join(log_dir,
                              config["file_name_prefix"] + "_run" + str(run_num) + "_vec_normalize_" + run.id + ".pkl")
    env.save(stats_path)
    return env


checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/',
                                         name_prefix=config["file_name_prefix"]+"_run"+str(run_num)+'_model_'+run.id)


# Create RL model
model = None
if config["rl_name"] == "PPO":
    model = PPO(config["policy_type"], make_env(), verbose=1,
                tensorboard_log="./tensorboard/"+config["file_name_prefix"]+"_run"+str(run_num)+"_llc_fiveJoints_"+run.id+"/",
                n_steps=config["n_steps"], batch_size=config["batch_size"])
elif config["rl_name"] == "A2C":
    model = A2C(config["policy_type"], make_env(), verbose=1,
                tensorboard_log="./tensorboard/", n_steps=config["n_steps"])
elif config["rl_name"] == "DQN":
    model = DQN(config["policy_type"], make_env(), learning_rate=config["learning_rate"] ,verbose=1,
                tensorboard_log="./tensorboard/"+config["file_name_prefix"]+"_run"+str(run_num)+"_llc_fiveJoints_"+run.id+"/", buffer_size=config["buffer_size"],
                target_update_interval=config["target_update_interval"],
                exploration_fraction=config["exploration_fraction"],
                learning_starts=config["learning_starts"], train_freq=config["train_freq"])

# target_update_interval : how many steps before we make Q_left = Q_right (could also try 200)
# take all the weights from network on the left, and copy it into network on the right (slowly or abruptly)

# try these next
# learning_starts=500
# model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/", buffer_size=6000, train_freq=(10, 'step'))
# model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/", buffer_size=4000)
# the smaller the buffer size, the more recent data is in buffer. Policy updates based on most recent experience./star
# if this makes no difference, leave it to 6000

# Train the agent
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=[WandbCallback(
        gradient_save_freq=100,
        model_save_path="models/" + config["file_name_prefix"] +"_run" + str(run_num) + "_llc_fiveJoints_"+run.id+"/visual_" + str({run.id}),
        verbose=2,
    ), checkpoint_callback],
)
# model.save("ppo2_bender")
# model.save("a2c_bender")
model.save(config["file_name_prefix"]+"_run"+str(run_num)+"_llc_fiveJoints_bender_"+run.id)

run.finish()

