import tianshou as ts
"""
Evaluate a trained policy
"""
def eval_policy(env, policy):
    c = ts.data.Collector(policy, env, ts.data.ReplayBuffer(size=100_000))
    c.collect(n_episode=1)
    actions = c.buffer[:len(c.buffer)].act.tolist()
    rewards = c.buffer[:len(c.buffer)].info['reward'].tolist()
    costs = c.buffer[:len(c.buffer)].info['cost'].tolist()
    
    return actions, rewards, costs