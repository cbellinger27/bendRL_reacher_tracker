import gym
import time
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os
# from stable_baselines3.common.evaluation import evaluate_policy
# from CartesianReacherFiveJointsGoal import CartesianReacherFiveJoints
from VisualReacherFiveJointsImageSpace import ReacherFiveJointsImageSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 20000,
    "learning_rate": 0.0001,
    "learning_starts": 200,
    "buffer_size": 1000,
    "target_update_interval": 200,
    "train_freq": 4,
    "exploration_fraction": 0.9,
    "env_name": "BenderFiveJoints",
    "rl_name": "PPO",
    "goal_threshold": 0.2,
    "random_start": 0,
    "file_name_prefix": "ppo_test",
    "n_steps": 256, # this is used with PPO and A2C,
    "batch_size": 32 # this is used with PPO
}
# log_dir = "./ppo_testrandom_start_run0_CartesianReacherLLC_690dwisu"
log_dir = "./ppo_testrandom_start_run0_ReacherFiveJointsImageSpaceLLC_3kz8yxog"
# stats_path = log_dir+"/ppo_test_run0_vec_normalize_690dwisu.pkl"

stats_path = "./ppo_smaller_threshold_point2_run0_llc_fiveJoints_3kz8yxog/ppo_smaller_threshold_point2_run0_vec_normalize_3kz8yxog.pkl"

def make_env():
    # env = DummyVecEnv([lambda: Monitor(CartesianReacherFiveJoints(random_start=0, log_state_actions=False,
    #                                                     goal_threshold=config["goal_threshold"]),
    #                                    log_dir)])
    # env = DummyVecEnv([lambda: Monitor(ReacherFiveJointsImageSpace(random_start=0, log_state_actions=False),log_dir)])
    env = DummyVecEnv([lambda: ReacherFiveJointsImageSpace(random_start=0, log_state_actions=False)])
    env = VecNormalize.load(stats_path, env)
    # env.training = False
    return env

env = make_env()


# model = PPO.load("/home/lamarchecl/git/visual_bender/src/models/ppo_test_run0_llc_fiveJoints_690dwisu/690dwisu/model", env)
model = PPO.load("ppo_smaller_threshold_point2_run0_llc_fiveJoints_bender_3kz8yxog")



# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
# print("the mean reward is: " + str(mean_reward))

obs = env.reset()
# env.render()
# input('Press Enter')
for _ in range(3):
    done = False
    while not done:
        # action, _states = model.predict(obs, deterministic=False)
        action = env.action_space.sample()
        print(action)
        obs, reward, done, info = env.step([action])
        print(reward)
        env.render()
    # time.sleep(0.2)

# TODO Update reward scheme
# Note,  when I put it to deterministic=False it gets to the goal but not directly.  Deterministic true fails

