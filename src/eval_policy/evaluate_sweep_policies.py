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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import yaml


# sweep_id = "sweep-i33pfy7g" # "334syzp0"  "ih0q9cko" "vuld4dog", "pw7kjclx"

sweep_id = "sweep-i33pfy7g"

def evaluate_policy(config, run_id):
    print("evaluating: " + sweep_id + ", " + run_id)
    start_pos_string = ["fixed_start"]
    norm_file = "run" + str(config["run"]['value']) + "_vec_normalize_" + run_id + ".pkl"
    log_dir = "/" + config["env_type"]['value'] + "/" + config["rl_name"]['value'] + "/" + \
              start_pos_string[config["random_start"]['value']] + "/" + "run" + \
              str(config["run"]['value']) + "_" + run_id
    stats_path = "" + log_dir + "/" + norm_file
    model_path = "/final_model." + log_dir
    run_details = [sweep_id, run_id, config['env_name']['value'], config['env_type']['value'], config['init_learning_rate']['value'], config['learning_rate']['value'], config['linear_scheduler']['value'], config['max_ep_len']['value'],  config['n_epochs']['value'], config['n_steps']['value'], config['policy_type']['value'], config['random_start']['value'], config['rl_name']['value'], config['run']['value'], config['shape_reward']['value'], config['circle_rad_importance']['value'], config['total_timesteps']['value']]
    pos = list(map(int, config['target_position']['value'].split(',')))
    j = 0
    e = 1
    sum_rewards = 0
    steps = []
    episodes = []
    rewards = []
    goals = []

    if not os.path.exists("/home/colin/Documents/repositories/visual_bender/src/"+stats_path):
        del pos 
        return np.vstack([steps, episodes, rewards, goals]).T, run_details
    
    env = DummyVecEnv([lambda: ReacherFiveJointsImageSpace(random_start=config['random_start']['value'],
                                                        target_position=pos,
                                                        STEPS_IN_EPISODE=config['max_ep_len']['value'],
                                                        save_images=False,
                                                        rad_imp=config['circle_rad_importance']['value'],
                                                        shape_reward=config['shape_reward']['value'],
                                                        render_mode="human",
                                                        env_type=config['env_type']['value'],
                                                        log_state_actions=False)])

    env = VecNormalize.load("/home/colin/Documents/repositories/visual_bender/src"+stats_path, env)
    #don't update the normalization parameters during evaluation
    # env.training = False
    #don't normalize evaluation rewards
    env.norm_reward = False

    model = None
    if config["rl_name"]['value'] == "DQN":
        model = DQN.load("/home/colin/Documents/repositories/visual_bender/src/"+model_path)
    if config["rl_name"]['value'] == "PPO":
        model = PPO.load("/home/colin/Documents/repositories/visual_bender/src/"+model_path)

    # mean_reward, std_reward = evaluate_policy(model, n_eval_episodes=5)
    # print("the mean reward is: " + str(mean_reward))
    obs = env.reset()
    for i in range(500):
        steps += [i+1]
        episodes += [e]
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        print(reward)
        rewards += reward.tolist()
        # print(obs)
        # print("step " + str(j) + " action " + str(action) + " reward " + str(reward))
        j += 1
        sum_rewards += reward
        at_goal = 0
        if done:
            e += 1
            if reward == 20:
                print("GOAL ATTAINED")
                at_goal = 1
            else:
                print("Too many steps or protective stop...")
            print("The sum of rewards is: % 0.3f" % sum_rewards)
            sum_rewards = 0
            # obs = env.reset()
            j = 0
        goals += [at_goal]
    env.close()
    del pos 
    return np.vstack([steps, episodes, rewards, goals]).T, run_details


#get each config in sweep directory
def main():
    dir = "/home/colin/Documents/repositories/visual_bender/src/wandb/"
    headings = []
    results = None
    details = [['sweep_id','run_id','env_name','env_type','init_learning_rate','learning_rate','linear_scheduler','max_ep_len','n_epochs','n_steps','policy_type','random_start','rl_name','run','shape_reward','circle_rad_importance','total_timesteps']]
    run_it = 1
    for filename in os.listdir(dir+sweep_id):
        run_id = str.split(filename, "config-")[1]
        run_id = str.split(run_id, '.yaml')[0]
        headings += ["Steps", "Episodes", "Rewards_" + run_id, "Goals_" + run_id]
        f = os.path.join(dir+sweep_id, filename)
        print(f)
        #verify that this is a file in the directory
        if os.path.isfile(f):
            with open(f, 'r') as file:
                config = yaml.safe_load(file)
                #evaluate the policy
                res, dets = evaluate_policy(config, run_id)
                if len(res) > 0:
                    details += [dets]
                    if results is None:
                        results = res
                    else:
                        print(results.shape)
                        print(res.shape)
                        results = np.concatenate((results, res),axis=1)
                        print(results.shape)
    print(results.shape)
    print(len(headings))
    resultsDF = pd.DataFrame(data=results, columns=headings)
    resultsDF.to_csv("eval_results_sweep_" + run_id + ".csv")
    print(details)
    np.vstack(details)
    np.savetxt("eval_details_sweep_" + run_id + ".csv", np.vstack(details), fmt="%s", delimiter=',')



if __name__ == "__main__":
    main()