import torch
import torch.nn.functional as F
from torch.distributions import Normal

class SACPolicyNet(torch.nn.Module):
    """SAC Policy NN for continuous action space
    Args:
     -state_dim, hidden_dim, action_dim
     -action_bound: boundary of continuous action space
    """
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(SACPolicyNet, self).__init__()
        
        self.fc = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        """
        Returns:
         -action: resampled action vector
         -log_prob: log probability of resampled action
          -if there Multidimensional action space, 
           then creat gassuian for every action
           example: action[1, 1, 1, 1] --> log_prob[0.3, 1.1, 0.5, 0.6]
        """
        x = F.relu(self.fc(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        # creat Gaussian distribution and resample action
        gauss_distri = Normal(mu, std)
        normal_sample = gauss_distri.rsample()
        # get log probability for gradiant and action for Q
        log_prob = gauss_distri.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob

class SACQvalueNet(torch.nn.Module):
    """SAC Critic NN
    Output:
     -Q value -- (batch_size, 1)
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(SACQvalueNet, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim+action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        """ Return: Q value """
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SACPolicyNetDiscrete(torch.nn.Module):
    """SAC Policy NN for dicscrete action space
    Args:
     -state_dim, hidden_dim, action_dim
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(SACPolicyNetDiscrete, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

class SACQvalueNetDiscrete(torch.nn.Module):
    """ SAC critic NN for dicscrete action space
    Output:
     -Q values -- (batch_size, action_dim)
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(SACQvalueNetDiscrete, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)





