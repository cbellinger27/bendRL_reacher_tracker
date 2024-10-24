import sys
sys.path.append("src/") 
import gym
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
from VisualReacherFiveJointsImageSpace import ReacherFiveJointsImageSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# sweep_configuration = {
#     "project": "ReacherFiveJointsImageSpace",
#     "entity": "nrc-rl-robotics",
#     "method": "grid",
#     "policy_type": "MlpPolicy",
#     "total_timesteps": 10000,
#     "learning_rate": 0.0003,
#     "learning_starts":  500,
#     "buffer_size": 3000,
#     "target_update_interval": 200,
#     "train_freq": 4,
#     "exploration_fraction": 0.99,
#     "env_name": "ReacherFiveJointsImageSpace",
#     "env_type": "reaching",
#     "rl_name": "PPO",
#     "n_steps": 512,
#     "batch_size": 64,
#     "shape_reward": 1,
#     "run": 2,
#     "random_start": 0,
#     "render_mode": "human",
#     "seed": 1234
# }

# "values": ["1700,900", "200,200", "200,900", "1700,200"]
sweep_configuration = {
    "project": "ReacherFiveJointsImageSpace",
    "entity": "nrc-rl-robotics",
    "env_name": "ReacherFiveJointsImageSpace",
    "target_position": "950,550",
    "random_start": 0,
    "env_type": "tracking",
    "method": "grid",
    "run": 3,
    "policy_type": "CnnPolicy",
    "max_ep_len": 200,
    "rl_name": "DQN",
    "render_mode": "human",
    "seed": 1234
}

start_pos_string = ["fixed_start"]
run_id = "1nakd1gz"
norm_file = "run"+str(sweep_configuration["run"])+"_vec_normalize_"+run_id+".pkl"
log_dir = sweep_configuration["env_type"] + "/" + sweep_configuration["rl_name"] + "/" + \
              start_pos_string[sweep_configuration["random_start"]] + "/" + "run" + \
              str(sweep_configuration["run"]) + "_" + run_id
# log_dir = "/" + sweep_configuration["env_type"] + "/" + sweep_configuration["rl_name"] + "/" + \
#               start_pos_string[sweep_configuration["random_start"]] + "/" + "run" + \
#               str(sweep_configuration["run"]) + "_" + run_id
stats_path = log_dir + "/" + norm_file
# stats_path = "./src" + log_dir + "/" + norm_file
# model_path = "./models/" + sweep_configuration["env_type"] + "/" + sweep_configuration["rl_name"] + "/" + \
#               start_pos_string[sweep_configuration["random_start"]] + "/" + "run" + \
#               str(sweep_configuration["run"]) + "_" + run_id + "/model.zip"

# model_path = "./src" + log_dir + "/checkpoint_run1_"+run_id+"_20000_steps"

model_path = "final_model./" + log_dir
# model_path = "static/DQN/fixed_start/run1_c135u79m/checkpoint_run1_c135u79m_40000_steps"
# model_path = "./src/final_model." + log_dir
# sweep_configuration["env_type"]
def make_env():
    pos = list(map(int, sweep_configuration['target_position'].split(',')))
    env = DummyVecEnv([lambda: ReacherFiveJointsImageSpace(random_start=0, STEPS_IN_EPISODE=sweep_configuration["max_ep_len"], save_images=True,
                                                           rad_imp=0.99, shape_reward=1,   render_mode="red_channel", target_position=pos,
                                                           env_type=sweep_configuration["env_type"], log_state_actions=False)])
    env = VecNormalize.load(stats_path, env)
    # env = VecNormalize(env, norm_obs=False, norm_reward=True)
    # env.training = True
    return env


env = make_env()
# env = BenderFourJoints(random_start=0, log_state_actions=True)
model = None
if sweep_configuration["rl_name"] == "DQN":
    model = DQN.load(model_path, env)
    # model = DQN.load("dqn_threshold_point2_run0_llc_fiveJoints_bender_nxw6bqk6", env)  # could replace this with the w&b saved model, or the checkpoint saved model
if sweep_configuration["rl_name"] == "PPO":
    model = PPO.load(model_path)

print(model)
# mean_reward, std_reward = evaluate_policy(model, n_eval_episodes=5)
# print("the mean reward is: " + str(mean_reward))
obs = env.reset()
j = 0
sum_rewards = 0
for i in range(800):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    # print(obs)
    # print("step " + str(j) + " action " + str(action) + " reward " + str(reward))
    j += 1
    sum_rewards += reward
    if done:
        if reward == 1:
            print("GOAL ATTAINED")
        else:
            print("Too many steps or protective stop...")
        print("The sum of rewards is: % 0.3f" % sum_rewards)
        sum_rewards = 0
        # obs = env.reset()
        j = 0


