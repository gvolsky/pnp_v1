import numpy as np
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def prepare_state(state):
    """
    Take state and return three masks: barrier positions, 
    preys positions and predator positions
    """
    barrier_mask = np.logical_and(state[:, :, 0] == -1, state[:, :, 1] == -1)
    preys_mask = state[:, :, 0] > 0
    predators_mask = np.logical_and(state[:, :, 0] == 0, state[:, :, 1] >= 0)
    return barrier_mask, preys_mask, predators_mask

def roll(arr, x, y):
    """
    arr is [barrier_mask, preys_mask, predators_mask]
    """
    height, width = arr.shape[1], arr.shape[2]
    dx = width // 2 - x
    dy = height // 2 - y
    res = np.roll(a=arr, shift=(dx, dy), axis=(2, 1))
    return res

def get_reward(
        info,
        step_reward=0.1,
        step_coef=0.99, 
        kill_reward=1.,
        kill_coef=1.01
        ):
    step = step_coef ** info['step'] * (np.array(info['true_action']) * step_reward)
    kill = kill_coef ** info['all_eated'] * (kill_reward * np.array(info['eated_on_step']))
    return step + kill

def Path(obs, info, acs, next_obs, next_info, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    return {"observation": np.array(obs, dtype=np.float32),
            "info": info,
            "action": np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "next_info": next_info,
            "terminal": np.array(terminals, dtype=np.float32)}

def sample_trajectory(env, agent):

    ob, ab = env.reset()
    agent.reset(ob, 0)
    done = False
    obs, info, acs, next_obs, next_info, terminals = [], [], [], [], [], []

    while not done:
        obs.append(ob)
        info.append(ab)
        ac = agent.get_actions(ob, 0)
        acs.append(ac)

        # take that action and record results
        ob, done, ab = env.step(ac)

        # record result of taking that action
        next_obs.append(ob)
        next_info.append(ab)
        terminals.append(done)


    return Path(obs, info, acs, next_obs, next_info, terminals)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)