import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
import gymnasium as gym
import json
import torch
import random
import numpy as np
from trainer import Trainer
from agent.policy import SACForDiscrete, SACForContinuous


class SACTrain:
    
    def __init__(self, env):
        with open('config.json', 'r') as file:
            config = json.load(file)[env]

        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.alpha_lr = config['alpha_lr']
        self.num_episodes = config['num_episodes']
        self.hidden_dim = config['hidden_dim']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.buffer_size = config['buffer_size']
        self.minimal_size = config['minimal_size']
        self.batch_size = config['batch_size']
        self.target_entropy = config['target_entropy']
        self.env_settings = config['env_settings']
        self.seed_config = config['seed']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.set_seeds()

        self.env = gym.make(env, **self.env_settings)

        self.reply_buffer = utils.ReplayBuffer(self.buffer_size)
        

    def set_seeds(self):
        random.seed(self.seed_config['random_seed'])
        np.random.seed(self.seed_config['numpy_seed'])
        torch.manual_seed(self.seed_config['torch_seed'])

    def learn_discrete(self, num_episodes=None):
        """
        Learning process for discrete environment
        """
        if num_episodes is None:
            num_episodes = self.num_episodes

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        agent = SACForDiscrete(state_dim, self.hidden_dim, action_dim, 
                            self.actor_lr, self.critic_lr, self.target_entropy, 
                            self.alpha_lr, self.gamma, self.tau, self.device)
        
        trainer = Trainer(self.env, agent, num_episodes,
                          self.reply_buffer, self.minimal_size, self.batch_size)
        return_list = trainer.discrete_train_off_policy_agent()
        return return_list, agent

    def learn_continuous(self, num_episodes=None, use_her=False):
        """
        Learning process for continuous environment
        """
        if num_episodes is None:
            num_episodes = self.num_episodes
            
        if use_her == True:
            self.reply_buffer = utils.HERReplayBufferTrajectory(self.buffer_size)
            
        state_dim = 12
        action_dim = self.env.action_space.shape[0]
        action_bound = self.env.action_space.high[0]
        
        self.target_entropy = -self.env.action_space.shape[0]

        agent = SACForContinuous(state_dim, 
                                 self.hidden_dim, 
                                 action_dim, 
                                 action_bound, 
                                 self.actor_lr, 
                                 self.critic_lr,
                                 self.target_entropy, 
                                 self.alpha_lr, 
                                 self.gamma, 
                                 self.tau, 
                                 self.device)
        
        trainer = Trainer(self.env, 
                          agent, 
                          num_episodes,
                          self.reply_buffer, 
                          self.minimal_size, 
                          self.batch_size)
        
        if use_her == True:
            return_list = trainer.continuous_train_off_policy_her(train_batch=self.num_episodes/100)
        else:
            return_list = trainer.continuous_train_off_policy()
            
        return return_list, agent




