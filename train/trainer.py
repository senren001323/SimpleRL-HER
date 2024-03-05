import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import utils
from tqdm import tqdm


class Trainer:

    def __init__(self, 
                 env, 
                 agent, 
                 num_episodes, 
                 replaybuffer, 
                 minimal_size, 
                 batch_size):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.replaybuffer = replaybuffer
        self.minimal_size = minimal_size
        self.batch_size = batch_size

    def continuous_train_off_policy(self):
        """
        Trainer for continuous panda-gym environment
        """
        return_list = []
        for i in range(10):
            with tqdm(total=int(self.num_episodes/10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes/10)):
                    episode_return = 0
                    observation, _ = self.env.reset()
                    state = utils.multi_input(observation)
                    done = False
                    while not done:
                        action = self.agent.take_action(state)
                        observation, reward, terminated, truncated, info = self.env.step(action)
                        next_state = utils.multi_input(observation)
                        done = terminated or truncated or info['is_success']
                        self.replaybuffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward
                        if self.replaybuffer.size() > self.minimal_size:
                            b_s, b_a, b_r, b_ns, b_dn = self.replaybuffer.sample(self.batch_size)
                            transition_dict = {'states': b_s, 'actions': b_a, 'rewards': b_r,
                                               'next_states': b_ns, 'dones': b_dn}
                            self.agent.update(transition_dict)
                    return_list.append(episode_return)
                    if (i_episode+1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (self.num_episodes/10 * i + i_episode+1),
                                          'return': '%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)
        return return_list

    def continuous_train_off_policy_her(self, train_batch):
        """Train policy use her
            and train with trajectory
        """
        return_list = []
        for i in range(10):
            with tqdm(total=int(self.num_episodes/10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes/10)):
                    episode_return = 0
                    observation, _ = self.env.reset()
                    state = utils.multi_input(observation)
                    trajectory = utils.Trajectory(state)
                    done = False
                    while not done:
                        action = self.agent.take_action(state)
                        observation, reward, terminated, truncated, info = self.env.step(action)
                        state = utils.multi_input(observation)
                        done = terminated or truncated or info['is_success']
                        trajectory.add_step(state, action, reward, done)
                        episode_return += reward
                    self.replaybuffer.add_trajectory(trajectory)
                    return_list.append(episode_return)
                    if self.replaybuffer.size() > self.minimal_size:
                        for _ in range(int(train_batch)):
                            transition_dict = self.replaybuffer.sample(self.batch_size)
                            self.agent.update(transition_dict)
                    if (i_episode+1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (self.num_episodes/10 * i + i_episode+1),
                                          'return': '%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)
        return return_list

    
    def discrete_train_off_policy_agent(self):
        """
        Trainer for discrete openai gym environment
        Discrete
        """
        return_list = []
        for i in range(10):
            with tqdm(total=int(self.num_episodes/10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes/10)):
                    episode_return = 0
                    state, _ = self.env.reset()
                    done = False
                    while not done:
                        action = self.agent.take_action(state)
                        next_state, reward, done, truncation, _ = self.env.step(action)
                        done = done or truncation
                        self.replaybuffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward
                        if self.replaybuffer.size() > self.minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = self.replaybuffer.sample(self.batch_size)
                            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns,
                                               'rewards': b_r, 'dones': b_d}
                            self.agent.update(transition_dict)
                    return_list.append(episode_return)
                    if (i_episode+1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (self.num_episodes/10 * i + i_episode+1),
                                          'return': '%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)
        return return_list
