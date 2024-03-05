import torch
from agent.net import *
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SACForContinuous:

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, 
                 actor_lr, critic_lr, target_entropy, alpha_lr, gamma, tau, device):
        # initialize actor and critic nets in continuous action space
        self.actor = SACPolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = SACQvalueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = SACQvalueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = SACQvalueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = SACQvalueNet(state_dim, hidden_dim, action_dim).to(device)
        # copy paramters to target critic nets
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # constraint alpha optimization
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.target_entropy = target_entropy

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        return action.detach().numpy().squeeze(0) # to (action_dim) for panda-gym env

    def cal_next_qvalue(self, next_state, reward, done):
        """Calculate Q' value(r + γV')

        Args:
         -next_state, reward: From replybuffer
         -done: Whether in terminal state
        Returns:
         -td target: Q' value -- (batch_size, 1)
        """
        next_action, next_log_prob = self.actor(next_state)
        next_entropy = -torch.sum(next_log_prob, dim=1, keepdim=True) # calculate total entropy in Joint action space
        target_qvalue_1 = self.target_critic_1(next_state, next_action)
        target_qvalue_2 = self.target_critic_2(next_state, next_action)
        target_qvalue = torch.min(target_qvalue_1, target_qvalue_2)
        td_target = reward + self.gamma * (target_qvalue + self.log_alpha.exp() * next_entropy) * (1-done)
        return td_target

    def soft_update(self, target_net, net):
        """Soft update target net

        Args:
         -target_net: Target network
         -net: Original network
        """
        for target_param, param in zip(target_net.parameters(),
                                       net.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

    def update(self, transition_dict):
        """Gradient update for Actor and Critic network

        Args:
         -transition_dict: A dictionary including s, a, r, s'
        """
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(self.device)
        actions = torch.tensor(np.stack(transition_dict["actions"]), dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float).view(-1, 1).to(self.device)
        # update critic net
        td_target = self.cal_next_qvalue(next_states, rewards, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach())
        )
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach())
        )
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        # update actor net
        new_actions, log_prob = self.actor(states)
        entropy = -torch.sum(log_prob, dim=1, keepdim=True)
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        q_value = torch.min(q1_value, q2_value)
        actor_loss = torch.mean(
            -self.log_alpha.exp()*entropy - q_value
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # update alpha
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp()
        )
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward
        self.alpha_optimizer.step()
        # soft update target critic net
        self.soft_update(self.target_critic_1, self.critic_1)
        self.soft_update(self.target_critic_2, self.critic_2)

class SACForDiscrete:

    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, target_entropy, alpha_lr, gamma, tau, device):
        # initialize actor and critic nets in discrete action space
        self.actor = SACPolicyNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1 = SACQvalueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = SACQvalueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = SACQvalueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = SACQvalueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        # copy paramters to target critic nets
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr) 

        self.log_alpha = torch.tensor(np.log(0.1), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.target_entropy = target_entropy
        
    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_distri = torch.distributions.Categorical(probs)
        action = action_distri.sample()
        return action.item()

    def cal_next_qvalue(self, next_state, reward, done):
        """
        Return:
         -td_target: r + γE(Qmin - αlogΠ')
        """
        next_probs = self.actor(next_state)
        log_next_probs = torch.log(next_probs + 1e-8)
        # directly calculate entropy
        next_entropy = -torch.sum(next_probs*log_next_probs, dim=1, keepdim=True)
        next_q1_value = self.target_critic_1(next_state)
        next_q2_value = self.target_critic_2(next_state)
        target_value = torch.sum(next_probs*torch.min(next_q1_value, next_q2_value), dim=1, keepdim=True)
        td_target = reward + self.gamma * (target_value + self.log_alpha.exp()*next_entropy) * (1-done)
        return td_target

    def soft_update(self, target_net, net):
        for target_param, param in zip(target_net.parameters(),
                                       net.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)#discrete space does not need float dtype
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float).view(-1, 1).to(self.device)
        # update critic nets
        td_target = self.cal_next_qvalue(next_states, rewards, dones)
        # extract value corresponding to each action
        critic_1_qvalues = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_qvalues, td_target.detach())
        )
        critic_2_qvalues = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_qvalues, td_target.detach())
        )
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        # update actor net
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        q_value = torch.sum(probs*torch.min(q1_value, q2_value), dim=1, keepdim=True)
        actor_loss = torch.mean(
            -self.log_alpha.exp()*entropy - q_value
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # update alpha
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp()
        )
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward
        self.alpha_optimizer.step()
        # soft update target critic net
        self.soft_update(self.target_critic_1, self.critic_1)
        self.soft_update(self.target_critic_2, self.critic_2)
        





