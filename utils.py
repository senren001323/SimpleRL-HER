import torch
import numpy as np
import collections
import random


class ReplayBuffer:
    """ReplyBuffer

    Buffer: (state, action, reward, next_state, done)
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Return touples of states, actions, next_states, rewards, dones 
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

class Trajectory:

    def __init__(self, reset_state):
        self.states = [reset_state]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.length = 0

    def add_step(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length += 1

class HERReplayBufferTrajectory:

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add_trajectory(self, trajectory):
        self.buffer.append(trajectory)

    def sample(self, batch_size, her_therashold=0.02, her_ratio=0.8):
        transition_dict = dict(
            states=[],
            actions=[],
            rewards = [],
            next_states = [],
            dones = [])
        for _ in range(batch_size):
            trajectory = random.sample(self.buffer, 1)[0]
            index_state = np.random.randint(trajectory.length)
            state = trajectory.states[index_state]
            action = trajectory.actions[index_state]
            reward = trajectory.rewards[index_state]
            next_state = trajectory.states[index_state+1]
            done = trajectory.dones[index_state]
            
            if np.random.uniform() <= her_ratio:
                index_goal = np.random.randint(index_state+1, trajectory.length+1)
                goal = trajectory.states[index_goal][-6:-3]
                dis = np.linalg.norm(next_state[-6:-3] - goal)
                done = False if dis > her_therashold else True
                if done:
                    reward = -0.5
                else:
                    reward = -1
                state = np.hstack((state[:-3], goal))
                next_state = np.hstack((next_state[:-3], goal))   
                
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)
            transition_dict['next_states'].append(next_state)
            transition_dict['dones'].append(done)

        transition_dict['states'] = np.array(transition_dict['states'])
        transition_dict['next_states'] = np.array(transition_dict['next_states'])
        transition_dict['actions'] = np.array(transition_dict['actions'])
        return transition_dict

    def size(self):
        return len(self.buffer)

def multi_input(ori_input):
    """Concat all vectors present if the input is a dictionary."""
    combined_array = np.concatenate(
        [value for value in ori_input.values()]
    )
    return combined_array

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))








